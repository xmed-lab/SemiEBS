import os
import sys
import logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='cps')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_20p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_80p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True) # <--
parser.add_argument('-ep', '--max_epoch', type=int, default=500)
parser.add_argument('--cps_loss', type=str, default='w_ce+dice')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--stu_w', type=float, default=1)
parser.add_argument('-s', '--ema_w', type=float, default=0.99)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=False) # <--
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None) # 200
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import math
import numbers
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import SimpleITK as sitk
from DiffVNet.diff_vnet import DiffVNet
from utils import EMA, maybe_mkdir, get_lr, fetch_data, fetch_data_aug, seed_worker, poly_lr, print_func, xavier_normal_init_weight, kaiming_normal_init_weight
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.data_loaders import KneeMRI, KneeMRI_light
from utils import config
from data.StrongAug import get_StrongAug
from data.transforms import CenterCrop, ToTensor



def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_stu_weight(epoch):
    if args.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epoch
        return args.stu_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.stu_w


# def get_current_ema_weight(epoch):
#     if args.cps_rampup:
#         # Consistency ramp-up from https://arxiv.org/abs/1610.02242
#         if args.consistency_rampup is None:
#             args.consistency_rampup = args.max_epoch
#         return args.ema_w * sigmoid_rampup(epoch, args.consistency_rampup)
#     else:
#         return args.ema_w


def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)

def make_loader(split, dst_cls=KneeMRI, repeat=None, is_training=True, unlabeled=False, raw=False, transforms_tr=None, transforms_val=None):
    if is_training:
        dst = dst_cls(
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            transform=transforms_tr
        )
        return DataLoader(
            dst,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker
        )
    else:
        dst = dst_cls(
            split=split,
            transform=transforms_val
        )
        return DataLoader(dst, pin_memory=True)


def make_model_all():
    model = DiffVNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )

    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=args.base_lr,
    #     weight_decay=1e-4
    # )


    return model, optimizer




class Difficulty:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(self.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights

    def cal_weights(self, pred,  label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_dice = self.dice_func(x_onehot, y_onehot, is_training=False)
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice>0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice<=0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)

        self.last_dice = cur_dice

        self.cls_learn = EMA(cur_cls_learn, self.cls_learn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)

        # print("---")
        # print(cur_diff.max(), cur_diff.min())

        cur_diff = torch.pow(cur_diff, 1/5)
        # cur_diff = cur_diff / cur_diff.max()
        # cur_diff = (cur_diff - cur_diff.min()) / (cur_diff.max() - cur_diff.min())

        # print(cur_diff.max(), cur_diff.min())

        self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        weights = cur_diff * self.dice_weight
        weights = weights / weights.max()
        return weights * self.num_cls





class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=3):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp((-((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.cuda(), groups=self.groups, padding="same")



if __name__ == '__main__':
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    vis_path = os.path.join(snapshot_path, 'vis')
    maybe_mkdir(vis_path)

    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S', force=True
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # make data loader
    transforms_train_labeled = get_StrongAug(config.patch_size, 3)
    transforms_train_unlabeled = get_StrongAug(config.patch_size, 1)
    transforms_val = transforms.Compose([CenterCrop(config.patch_size),ToTensor()])


    unlabeled_loader = make_loader(args.split_unlabeled, dst_cls=KneeMRI_light, unlabeled=True, transforms_tr=transforms_train_unlabeled)
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset), transforms_tr=transforms_train_labeled)
    eval_loader = make_loader(args.split_eval, is_training=False, transforms_val=transforms_val)



    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    model, optimizer = make_model_all()
    # model = kaiming_normal_init_weight(model)

    diff = Difficulty(config.num_cls, accumulate_iters=50)
    deno_loss  = make_loss_function(args.sup_loss)
    sup_loss  = make_loss_function(args.sup_loss)
    unsup_loss1  = make_loss_function(args.cps_loss)


    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    # ema_w = get_current_ema_weight(0)
    stu_w = get_current_stu_weight(0)
    best_eval = 0.0
    best_epoch = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_cps_list = []
        loss_sup_list = []
        supervised_diff_list = []
        supervised_supp_list = []
        supervised_easy_list = []


        model.train()
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):

            for s_name, s_params in model.decoder_stu.named_parameters():
                if s_name in model.denoise_model.decoder.state_dict().keys():
                    d_params = model.denoise_model.decoder.state_dict()[s_name]
                    t_params = model.decoder_t1.state_dict()[s_name]
                    if s_params.shape == d_params.shape:
                        s_params.data = args.ema_w * s_params.data + (1 - args.ema_w) * (d_params.data + t_params.data) / 2.0



            optimizer.zero_grad()
            image_l, label_l = fetch_data(batch_l)
            label_l = label_l.long()

            image_u = fetch_data(batch_u, labeled=False)

            if args.mixed_precision:
                with autocast():
                    shp = (args.batch_size, config.num_cls)+config.patch_size

                    label_l_onehot = torch.zeros(shp).cuda()
                    label_l_onehot.scatter_(1, label_l, 1)

                    x_start = label_l_onehot * 2 - 1

                    x_t, t, noise = model(x=x_start, pred_type="q_sample")
                    # print(t)


                    pred_xstart = model(x=x_t, step=t, image=image_l, pred_type="denoise")
                    output_l_t1 = model(image=image_l, pred_type="teacher_1")

                    # if epoch_num % 10 ==0:
                        # save = sitk.GetImageFromArray(image_l[0][0].data.cpu().numpy().astype(np.float32))
                        # sitk.WriteImage(save, vis_path+f'/image_l.nii.gz')
                        # save = sitk.GetImageFromArray(label_l[0][0].data.cpu().numpy().astype(np.float32))
                        # sitk.WriteImage(save, vis_path+f'/label_l.nii.gz')
                        # for cls in range(pred_xstart.shape[1]):
                    #         save = sitk.GetImageFromArray(x_start[0][cls].data.cpu().numpy().astype(np.float32))
                    #         sitk.WriteImage(save, vis_path+f'/x_start_{cls}.nii.gz')
                    #
                    #         save = sitk.GetImageFromArray(x_t[0][cls].data.cpu().numpy().astype(np.float32))
                    #         sitk.WriteImage(save, vis_path+f'/x_t_{cls}.nii.gz')
                    #
                    #         save = sitk.GetImageFromArray(pred_xstart[0][cls].data.cpu().numpy().astype(np.float32))
                    #         sitk.WriteImage(save, vis_path+f'/pred_xstart_{cls}.nii.gz')


                    # print(pred_xstart.shape)
                    # loss_teacher = bce(pred_xstart, label_l_onehot) + dice_loss(pred_xstart, label_l_onehot)
                    loss_teacher = deno_loss(pred_xstart, label_l)

                    # diff_mask = diff.generate_entropy_diff_weight(pred_xstart.detach())
                    weight_diff = diff.cal_weights(pred_xstart.detach(), label_l)
                    # weight_easy = weight_diff.max() - weight_diff + weight_diff.min()

                    sup_loss.update_weight(weight_diff)

                    loss_teacher_1 = sup_loss(output_l_t1, label_l)



                    with torch.no_grad():
                        output_u = model(image_u, pred_type="ddim_sample")
                        # output_u_mask = output_u[output_u==1]
                        # print(output_u.shape)
                        output_u_t1 = model(image_u, pred_type="teacher_1")
                        smoothing = GaussianSmoothing(config.num_cls, 5, 1)
                        output_u = smoothing(F.gumbel_softmax(output_u, dim=1))
                        output_u_t1 = F.softmax(output_u_t1, dim=1)
                        # uncertainty = diff.generate_entropy_diff_mask(output_u)
                        # print(uncertainty.max(), uncertainty.min())
                        # output_u_t1 = output_u_t1 * uncertainty

                        # if epoch_num % 10 ==0:
                        #     save = sitk.GetImageFromArray(image_u[0][0].data.cpu().numpy().astype(np.float32))
                        #     sitk.WriteImage(save, vis_path+'/image_u.nii.gz')
                        #     # save = sitk.GetImageFromArray((image_u[0][1].data.cpu().numpy()*255).astype(np.uint8))
                        #     # sitk.WriteImage(save, '../image_u1.nii.gz')
                        #
                        #     for cls in range(output_u.shape[1]):
                        #         save = sitk.GetImageFromArray(output_u[0][cls].data.cpu().numpy().astype(np.float32))
                        #         sitk.WriteImage(save, vis_path+f'/ddim_sample_{cls}.nii.gz')
                        #         save = sitk.GetImageFromArray(output_u_t1[0][cls].data.cpu().numpy().astype(np.float32))
                        #         sitk.WriteImage(save, vis_path+f'/teacher_{cls}.nii.gz')

                        # output_u = torch.softmax(output_u, dim=1)
                        # output_u_t1 = torch.softmax(output_u_t1, dim=1)
                        pseudo_label = torch.argmax(output_u + output_u_t1, dim=1, keepdim=True)
                        # pseudo_label_onehot = torch.zeros(shp).cuda()
                        # pseudo_label_onehot.scatter_(1, pseudo_label, 1)
                    # output_dist_u = output_dist[tmp_bs:, ...]

                    # print(output_major_l.shape, output_diff_l.shape, output_dist_l.shape)


                    # x_start_stu = pseudo_label_onehot * image_u

                    # x_start_stu = (x_start_stu) * 2 - 1
                    # x_t_stu, t_stu, noise_stu = model(x=x_start_stu, pred_type="q_sample")

                    output_stu_u = model(image=image_u, pred_type="student")

                    loss_student = unsup_loss1(output_stu_u, pseudo_label.detach())
                    # loss_student_2 = unsup_loss2(output_stu_u, pseudo_label_t1.detach())

                    loss = loss_teacher + loss_teacher_1 + args.stu_w * loss_student

                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer)
                # amp_grad_scaler.step(optimizer_stu)
                # amp_grad_scaler.step(optimizer_supp)
                amp_grad_scaler.update()
                # if epoch_num % args.consistency_rampup == 100:



            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(loss_teacher.item())
            loss_cps_list.append(loss_student.item())
            supervised_diff_list.append(loss_teacher_1.item())
            # supervised_supp_list.append(supervised_supp.item())
            # supervised_easy_list.append(supervised_easy.item())


        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        # writer.add_scalar('ema_w', ema_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/deno', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/stu', np.mean(loss_cps_list), epoch_num)
        writer.add_scalar('loss/diff', np.mean(supervised_diff_list), epoch_num)
        # writer.add_scalar('loss/supp', np.mean(supervised_supp_list), epoch_num)
        # writer.add_scalar('loss/easy', np.mean(supervised_easy_list), epoch_num)



        # print(dict(zip([i for i in range(config.num_cls)] ,print_func(weight_A))))
        writer.add_scalars('class_weights/A', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_diff))), epoch_num)
        # writer.add_scalars('class_weights/B', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_easy))), epoch_num)
        # writer.add_scalars('class_weights/C', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_supp))), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)} | stu_w: {stu_w}')
        # logging.info(f'     cls_indicator: {print_func(weights)}')
        # if epoch_num>0:
        logging.info(f"     diff_w: {print_func(weight_diff)}")
        # logging.info(f"     easy_w: {print_func(weight_easy)}")
        # logging.info(f"     supp_w: {print_func(weight_supp)}")
        # lr_scheduler_A.step()
        # lr_scheduler_B.step()
        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        # optimizer_stu.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)


        # print(optimizer_A.param_groups[0]['lr'])
        # ema_w = get_current_ema_weight(epoch_num)
        stu_w = get_current_stu_weight(epoch_num)

        if epoch_num % 1 == 0:

            # ''' ===== evaluation
            dice_list = [[] for _ in range(config.num_cls-1)]
            model.eval()
            dice_func = SoftDiceLoss(smooth=1e-8, do_bg=False)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    # print(image.shape)
                    # output_stu = model(image, pred_type="ddim_sample", branch="stu")
                    output_stu = model(image, pred_type="student")
                    save = sitk.GetImageFromArray(image[0][0].data.cpu().numpy().astype(np.float32))
                    sitk.WriteImage(save, vis_path+f'/val_image.nii.gz')
                    # for cls in range(output_stu.shape[1]):
                    # save = sitk.GetImageFromArray(output_stu[0][cls].data.cpu().numpy().astype(np.float32))
                    # sitk.WriteImage(save, vis_path+f'/val_stu_{cls}.nii.gz')

                    # print(output_all.shape)
                    # print(output.shape)
                    del image

                    # output_stu = torch.sigmoid(output_stu)

                    # output_stu = (output_stu > 0.5).float()

                    # print(output_stu.shape)


                    shp = (output_stu.shape[0], config.num_cls) + output_stu.shape[2:]
                    gt = gt.long()
                    save = sitk.GetImageFromArray(gt[0][0].data.cpu().numpy().astype(np.float32))
                    sitk.WriteImage(save, vis_path+f'/val_gt.nii.gz')
                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    output_stu = torch.argmax(output_stu, dim=1, keepdim=True).long()
                    save = sitk.GetImageFromArray(output_stu[0][0].data.cpu().numpy().astype(np.float32))
                    sitk.WriteImage(save, vis_path+f'/val_pred.nii.gz')

                    x_onehot.scatter_(1, output_stu, 1)


                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            writer.add_scalar('val_dice', np.mean(dice_mean), epoch_num)
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            # '''
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                }, save_path)
                # torch.save(model.state_dict(), save_path)
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break

    writer.close()
