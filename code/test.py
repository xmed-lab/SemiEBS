import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='fully')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--speed', type=int, default=0)
parser.add_argument('-g', '--gpu', type=str,  default='0')
parser.add_argument('--cps', type=str, default=None)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from models.vnet import VNet, VNet4SSNet
from models.vnet_dst import VNet_Decoupled
from DiffVNet.diff_vnet import DiffVNet
from models.vnet_decouple import *
from models.unet_ds import unet_3D_ds
from models.unet import unet_3D
from utils import test_all_case, read_list, maybe_mkdir, test_all_case_AB, test_all_case_diff
from utils import config

if __name__ == '__main__':
    stride_dict = {
        0: (16, 8),   # 1.5h
        1: (64, 48),  # 20min
        2: (128, 32), # 10min
        3: (128, 64),
    }
    stride = stride_dict[args.speed]

    snapshot_path = f'./logs/{args.exp}/'
    test_save_path = f'./logs/{args.exp}/predictions_{args.cps}/'
    maybe_mkdir(test_save_path)




    if "dst" in args.exp:
        if args.cps == "AB":
            model_A = VNet_Decoupled(
                n_channels=config.num_channels,
                n_classes=config.num_cls,
                n_filters=config.n_filters,
                normalization='batchnorm',
                has_dropout=False
            ).cuda()
            model_B = VNet_Decoupled(
                n_channels=config.num_channels,
                n_classes=config.num_cls,
                n_filters=config.n_filters,
                normalization='batchnorm',
                has_dropout=False
            ).cuda()
            model_A.eval()
            model_B.eval()
        else:
            model = VNet_Decoupled(
                n_channels=config.num_channels,
                n_classes=config.num_cls,
                n_filters=config.n_filters,
                normalization='batchnorm',
                has_dropout=False
            ).cuda()
            model.eval()

    elif "diffusion" in args.exp:
        # model = DiffUNet().cuda()
        model = DiffVNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        args.cps = "diffusion"

    elif "urpc" in args.exp:
        model = unet_3D_ds(n_classes=config.num_cls, in_channels=1).cuda()
        model.eval()
        args.cps = None
    # elif "acisis" in args.exp:
    #     model = unet_3D(n_classes=config.num_cls, in_channels=1).cuda()
    #     model.eval()
    #     args.cps = None
    elif "uamt" in args.exp or "acisis" in args.exp:
        model = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model.eval()
        args.cps = None
    elif "ssnet" in args.exp:
        model = VNet4SSNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False).cuda()
        model.eval()
        args.cps = None
    elif "dhc_cl" in args.exp:
        model = VNet_Decouple_deep(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False).cuda()
        model.eval()
        args.cps = "ensemble"
    elif "dhc_dec" in args.exp:
        model = VNet_Decouple_deep(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False).cuda()
        model.eval()
        args.cps = "ensemble"
    elif "dhc_shallow" in args.exp:
        model = VNet_Decouple_deep(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False).cuda()
        model.eval()
        args.cps = "ensemble"
    elif "3teachers" in args.exp:
        model = VNet_3teachers(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False).cuda()
        model.eval()
        args.cps = "3teachers"
    else:
        if args.cps == "AB":
            if "diff_models" in args.exp:
                model_A = VNet(
                    n_channels=config.num_channels,
                    n_classes=config.num_cls,
                    n_filters=config.n_filters,
                    normalization='batchnorm',
                    has_dropout=False
                ).cuda()
                model_B = unet_3D(n_classes=config.num_cls, in_channels=1).cuda()
            elif "cps_unet" in args.exp:
                model_A = unet_3D(n_classes=config.num_cls, in_channels=1).cuda()
                model_B = unet_3D(n_classes=config.num_cls, in_channels=1).cuda()
            else:
                model_A = VNet(
                    n_channels=config.num_channels,
                    n_classes=config.num_cls,
                    n_filters=config.n_filters,
                    normalization='batchnorm',
                    has_dropout=False
                ).cuda()
                model_B = VNet(
                    n_channels=config.num_channels,
                    n_classes=config.num_cls,
                    n_filters=config.n_filters,
                    normalization='batchnorm',
                    has_dropout=False
                ).cuda()
            model_A.eval()
            model_B.eval()
        else:
            model = VNet(
                n_channels=config.num_channels,
                n_classes=config.num_cls,
                n_filters=config.n_filters,
                normalization='batchnorm',
                has_dropout=False
            ).cuda()
            model.eval()

    ckpt_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')



    with torch.no_grad():
        if args.cps == "AB":
            model_A.load_state_dict(torch.load(ckpt_path)["A"])
            model_B.load_state_dict(torch.load(ckpt_path)["B"])
            print(f'load checkpoint from {ckpt_path}')
            test_all_case_AB(
                model_A, model_B,
                read_list(args.split),
                num_classes=config.num_cls,
                patch_size=config.patch_size,
                stride_xy=stride[0],
                stride_z=stride[1],
                test_save_path=test_save_path
            )

        elif args.cps == "diffusion":
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])
            print(f'load checkpoint from {ckpt_path}')
            test_all_case_diff(
                model,
                read_list(args.split),
                num_classes=config.num_cls,
                patch_size=config.patch_size,
                stride_xy=stride[0],
                stride_z=stride[1],
                test_save_path=test_save_path
            )



        else:
            if args.cps:
                model.load_state_dict(torch.load(ckpt_path)[args.cps])
            else: # for full-supervision
                model.load_state_dict(torch.load(ckpt_path))
            print(f'load checkpoint from {ckpt_path}')
            test_all_case(
                model,
                read_list(args.split),
                num_classes=config.num_cls,
                patch_size=config.patch_size,
                stride_xy=stride[0],
                stride_z=stride[1],
                test_save_path=test_save_path
            )
