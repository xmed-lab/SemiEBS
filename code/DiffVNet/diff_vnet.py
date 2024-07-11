import torch
import torch.nn as nn
# from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
# from light_training.utils.files_helper import save_new_model_and_delete_last
# import argparse
# import yaml
# from .basic_unet import BasicUNetEncoder
from .basic_unet_denose import BasicUNetDe, BasicUNetDe_stu
from .guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from .guided_diffusion.respace import SpacedDiffusion, space_timesteps
from .guided_diffusion.resample import UniformSampler
import math


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


class ConvBlockOneStage(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlockOneStage, self).__init__()
        self.normalization = normalization
        self.ops = nn.ModuleList()
        self.conv = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        if normalization == 'batchnorm':
            self.norm = nn.BatchNorm3d(n_filters_out)
        elif normalization == 'groupnorm':
            self.norm = nn.GroupNorm(num_groups=16, num_channels=n_filters_out)
        elif normalization == 'instancenorm':
            self.norm = nn.InstanceNorm3d(n_filters_out)
        elif normalization != 'none':
            assert False

    def forward(self, x):

        conv = self.conv(x)
        # relu = self.relu(conv)
        if self.normalization != 'none':
            norm = self.norm(conv)
        else:
            norm = conv
        out = self.relu(norm)
        return out


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        self.ops = nn.ModuleList()
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            self.ops.append(ConvBlockOneStage(input_channel, n_filters_out, normalization))

    def forward(self, x):

        for layer in self.ops:
            x = layer(x)
        # x = self.conv(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
            ops.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            ops.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        x = self.ops(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
            ops.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            ops.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))

        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        x = self.ops(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_classes=2, n_filters=16, normalization='none', has_dropout=False, dropout_rate=0.5):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=dropout_rate, inplace=True)

    def forward(self, x1, x2, x3, x4, x5):
        # x1, x2, x3, x4, x5 = features
        # print(x1.size(), x2.size(), x3.size())
        x5_up = self.block_five_up(x5)
        # print("1",x5.size(), x5_up.size(), x4.size())
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)

        out = self.out_conv(x9)
        return out


class Encoder(nn.Module):
    def __init__(self, n_filters, normalization, has_dropout, dropout_rate):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        # self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.dropout = nn.Dropout3d(p=dropout_rate, inplace=True)

    def forward(self, input):
        # print(input.size())
        x1 = input
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        return x1, x2, x3, x4, x5




class TembFusion(nn.Module):
    def __init__(self, n_filters_out):
        super(TembFusion, self).__init__()
        self.temb_proj = torch.nn.Linear(512, n_filters_out)
    def forward(self, x, temb):
        if temb is not None:
            x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        else:
            x =x
        return x



class TimeStepEmbedding(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, normalization='none'):
        super(TimeStepEmbedding, self).__init__()

        self.embed_dim_in = 128
        self.embed_dim_out = self.embed_dim_in * 4

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.embed_dim_in,
                            self.embed_dim_out),
            torch.nn.Linear(self.embed_dim_out,
                            self.embed_dim_out),
        ])

        self.emb = ConvBlockTemb(1, n_filters_in, n_filters_out, normalization=normalization)

    def forward(self, x, t=None, embeddings=None, image=None):
        if t is not None:
            temb = get_timestep_embedding(t, self.embed_dim_in)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)

        else:
            temb = None

        # print(temb.shape)

        if image is not None :
            x = torch.cat([image, x], dim=1)
        # if temb is not None:
        x = self.emb(x, temb)
        # else:
        #     x = self.emb(x)
        # x = self.conv(x)
        return x, temb, embeddings






class ConvBlockTembOneStage(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, normalization='none', temb_flag=True):
        super(ConvBlockTembOneStage, self).__init__()

        self.temb_flag = temb_flag
        self.normalization = normalization

        self.ops = nn.ModuleList()

        self.conv = nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        if normalization == 'batchnorm':
            self.norm = nn.BatchNorm3d(n_filters_out)
        elif normalization == 'groupnorm':
            self.norm = nn.GroupNorm(num_groups=16, num_channels=n_filters_out)
        elif normalization == 'instancenorm':
            self.norm = nn.InstanceNorm3d(n_filters_out)
        elif normalization != 'none':
            assert False

        self.temb_fusion =  TembFusion(n_filters_out)


    def forward(self, x, temb):

        conv = self.conv(x)
        # relu = self.relu(conv)
        if self.normalization != 'none':
            norm = self.norm(conv)
        else:
            norm = conv
        relu = self.relu(norm)
        if self.temb_flag:
            out = self.temb_fusion(relu, temb)
        else:
            out = relu
        return out


class ConvBlockTemb(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlockTemb, self).__init__()

        self.ops = nn.ModuleList()
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            if i < n_stages-1:
                self.ops.append(ConvBlockTembOneStage(input_channel, n_filters_out, normalization))
            else:
                self.ops.append(ConvBlockTembOneStage(input_channel, n_filters_out, normalization, temb_flag=False))

        # print(self.ops)

        # self.conv = nn.Sequential(*ops)

    def forward(self, x, temb):

        for layer in self.ops:
            x = layer(x, temb)
        # x = self.conv(x)
        return x


class DownsamplingConvBlockTemb(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlockTemb, self).__init__()

        self.ops = nn.ModuleList()
        if normalization != 'none':
            self.ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                self.ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                self.ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                self.ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
            self.ops.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        else:
            self.ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            self.ops.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        self.ops.append(TembFusion(n_filters_out))

        # self.conv = nn.Sequential(*ops)

    def forward(self, x, temb):

        for layer in self.ops:
            if layer.__class__.__name__ == "TembFusion":
                x = layer(x, temb)
            else:
                x = layer(x)
        # x = self.conv(x)
        return x


class UpsamplingDeconvBlockTemb(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlockTemb, self).__init__()

        self.ops = nn.ModuleList()
        if normalization != 'none':
            self.ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                self.ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                self.ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                self.ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
            self.ops.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        else:
            self.ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            self.ops.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        self.ops.append(TembFusion(n_filters_out))

        # self.conv = nn.Sequential(*ops)

    def forward(self, x, temb):
        for layer in self.ops:
            if layer.__class__.__name__ == "TembFusion":
                x = layer(x, temb)
            else:
                x = layer(x)
        # x = self.conv(x)
        return x



class Encoder_denoise(nn.Module):
    def __init__(self, n_filters, normalization, has_dropout, dropout_rate):
        super(Encoder_denoise, self).__init__()
        self.has_dropout = has_dropout
        # self.block_one = ConvBlockTemb(1, n_channels, n_filters, normalization=normalization)
        # self.block_one = ConvBlockTemb(1, n_channels, n_filters, normalization=normalization)

        self.block_one_dw = DownsamplingConvBlockTemb(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlockTemb(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlockTemb(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlockTemb(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlockTemb(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlockTemb(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlockTemb(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlockTemb(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.dropout = nn.Dropout3d(p=dropout_rate, inplace=True)

    def forward(self, x, temb, embeddings=None):

        x1 = x

        if embeddings is not None:
            x1 += embeddings[0]



        # print(input.size())
        # x1 = self.block_one(input, temb)
        x1_dw = self.block_one_dw(x1, temb)

        x2 = self.block_two(x1_dw, temb)
        if embeddings is not None:
            x2 = x2 + embeddings[1]
        x2_dw = self.block_two_dw(x2, temb)

        x3 = self.block_three(x2_dw, temb)
        if embeddings is not None:
            x3 = x3 + embeddings[2]
        x3_dw = self.block_three_dw(x3, temb)

        x4 = self.block_four(x3_dw, temb)
        if embeddings is not None:
            x4 = x4 + embeddings[3]
        x4_dw = self.block_four_dw(x4, temb)

        x5 = self.block_five(x4_dw, temb)
        if self.has_dropout:
            x5 = self.dropout(x5)

        return x1, x2, x3, x4, x5, temb

class Decoder_denoise(nn.Module):
    def __init__(self, n_classes=2, n_filters=16, normalization='none', has_dropout=False, dropout_rate=0.5):
        super(Decoder_denoise, self).__init__()
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlockTemb(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlockTemb(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlockTemb(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlockTemb(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlockTemb(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlockTemb(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlockTemb(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlockTemb(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=dropout_rate, inplace=True)

    def forward(self, x1, x2, x3, x4, x5, temb):
        # x1, x2, x3, x4, x5 = features
        # print(x1.size(), x2.size(), x3.size())
        x5_up = self.block_five_up(x5, temb)
        # print("1",x5.size(), x5_up.size(), x4.size())
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up, temb)
        x6_up = self.block_six_up(x6, temb)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up, temb)
        x7_up = self.block_seven_up(x7, temb)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up, temb)
        x8_up = self.block_eight_up(x8, temb)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up, temb)
        if self.has_dropout:
            x9 = self.dropout(x9)

        out = self.out_conv(x9)
        return out


class DenoiseModel(nn.Module):
    def __init__(self, n_classes, n_channels, n_filters, normalization, has_dropout, dropout_rate):
        super().__init__()
        self.embedding_diffusion = TimeStepEmbedding(n_classes+n_channels, n_filters, normalization=normalization)
        self.encoder = Encoder_denoise(n_filters, normalization, has_dropout, dropout_rate)
        self.decoder = Decoder_denoise(n_classes, n_filters, normalization, has_dropout, dropout_rate)
        # self.decoder_t1 = Decoder_denoise(n_classes, n_filters, normalization, has_dropout)
        # self.decoder_t2 = Decoder_denoise(n_classes, n_filters, normalization, has_dropout)
        # self.decoder_stu = Decoder_denoise(n_classes, n_filters, normalization, has_dropout)


    def forward(self, x: torch.Tensor, t, embeddings=None, image=None, branch=None):

        x, temb, embeddings = self.embedding_diffusion(x, t=t, embeddings=embeddings, image=image)
        x1, x2, x3, x4, x5, temb = self.encoder(x, temb, embeddings=embeddings)
        # if branch == "t1":
        #     out = self.decoder_t1(x1, x2, x3, x4, x5, temb)
        # elif branch == "t2":
        #     out = self.decoder_t2(x1, x2, x3, x4, x5, temb)
        # elif branch == "stu":
        #     out = self.decoder_stu(x1, x2, x3, x4, x5, temb)
        # else:
        out = self.decoder(x1, x2, x3, x4, x5, temb)
        return out


class DiffVNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, dropout_rate=0.5):
        super(DiffVNet, self).__init__()
        self.has_dropout = has_dropout
        self.n_classes = n_classes
        self.time_steps = 1000

        self.embedding_img = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.embedding = TimeStepEmbedding(n_channels, n_filters, normalization=normalization)

        self.encoder_img = Encoder(n_filters, normalization, has_dropout, dropout_rate)
        self.decoder_stu = Decoder(n_classes, n_filters, normalization, has_dropout, dropout_rate)
        self.decoder_t1 = Decoder(n_classes, n_filters, normalization, has_dropout, dropout_rate)

        # self.decoder_3 = Decoder(n_classes, n_filters, normalization, has_dropout)
        # self.decoder_4 = Decoder(n_classes, n_filters, normalization, has_dropout)
        self.denoise_model = DenoiseModel(n_classes, n_channels, n_filters, normalization, has_dropout,dropout_rate)

        betas = get_named_beta_schedule("linear", self.time_steps)
        # print(betas)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(self.time_steps, [self.time_steps]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(self.time_steps, [10]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(self.time_steps)


    def forward(self, image=None, x=None, pred_type=None, step=None, branch=None):


        if pred_type == "q_sample":
            noise = torch.randn_like(x).cuda()
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            # print(image.type(), temb.type())
            embeddings = self.embedding_img(image)
            embeddings = self.encoder_img(embeddings)
            # list
            # print(len(list(embeddings)))
            x, temb, embeddings = self.denoise_model.embedding_diffusion(x, t=step, image=image, embeddings=embeddings)
            x1, x2, x3, x4, x5, temb = self.denoise_model.encoder(x, temb, embeddings=embeddings)
            return self.denoise_model.decoder(x1, x2, x3, x4, x5, temb)

        elif pred_type == "student":
            # x, temb, embeddings = self.embedding_stu(image)
            # temb=None
            # embeddings=None
            embeddings = self.embedding_img(image)
            embeddings = self.encoder_img(embeddings)
            x, temb, embeddings = self.embedding(image,embeddings=embeddings)
            x1, x2, x3, x4, x5, temb = self.denoise_model.encoder(x, temb, embeddings=embeddings)
            return self.decoder_stu(x1, x2, x3, x4, x5)

        elif pred_type == "teacher_1":
            embeddings = self.embedding_img(image)
            embeddings = self.encoder_img(embeddings)
            x, temb, embeddings = self.embedding(image, embeddings=embeddings)
            x1, x2, x3, x4, x5, temb = self.denoise_model.encoder(x, temb, embeddings=embeddings)
            return self.decoder_t1(x1, x2, x3, x4, x5)

        elif pred_type == "ddim_sample":
            # print((image.shape[0], self.n_classes) + image.shape[2:])

            # print(self.diffusion_model.forward)

            embeddings = self.embedding_img(image)
            embeddings = self.encoder_img(embeddings)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.denoise_model, (image.shape[0], self.n_classes) + image.shape[2:],
                                                                model_kwargs={"image": image, "embeddings":embeddings})
            sample_out = sample_out["pred_xstart"]

            # sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (2, 13, 64, 128, 128), model_kwargs={"image": image})
            # sample_out = sample_out["pred_xstart"]
            return sample_out
