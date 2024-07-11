import torch
from torch import nn



class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
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
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
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
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x



class Decoder(nn.Module):
    def __init__(self, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
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

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features
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
    def __init__(self, n_channels, n_filters, normalization, has_dropout):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        # print(input.size())
        x1 = self.block_one(input)
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

class VNet_Decouple_deep(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet_Decouple_deep, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.decoder_1 = Decoder(n_classes, n_filters, normalization, has_dropout)
        self.decoder_2 = Decoder(n_classes, n_filters, normalization, has_dropout)
        self.decoder_3 = Decoder(n_classes, n_filters, normalization, has_dropout)

        self.cls_indicator = torch.Tensor(n_classes).cuda()


        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def encoder(self, input):
        # print(input.size())
        x1 = self.block_one(input)
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



    def forward(self, image, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        
        features = self.encoder(image)
        out_org = self.decoder_1(features)
        out_diff = self.decoder_2(features)
        out_dist = self.decoder_3(features)
        
        if turnoff_drop:
            self.has_dropout = has_dropout
        
        return out_org, out_diff, out_dist



class VNet_Decouple(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet_Decouple, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_nine_diff = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_nine_dist = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv_diff = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv_dist = nn.Conv3d(n_filters, n_classes, 1, padding=0)


        self.cls_indicator = torch.Tensor(n_classes).cuda()

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def encoder(self, input):
        # print(input.size())
        x1 = self.block_one(input)
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

    def decoder(self, features):
        x1, x2, x3, x4, x5 = features
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
        x9_diff = self.block_nine_diff(x8_up)
        x9_dist = self.block_nine_dist(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
            x9_diff = self.dropout(x9_diff)
            x9_dist = self.dropout(x9_dist)

        return x9, x9_diff, x9_dist

    def forward(self, image, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        features = self.encoder(image)
        x9, x9_diff, x9_dist = self.decoder(features)

        out_org = self.out_conv(x9)
        out_diff = self.out_conv_diff(x9_diff)
        out_dist = self.out_conv_dist(x9_dist)

        if turnoff_drop:
            self.has_dropout = has_dropout

        return out_org, out_diff, out_dist


class VNet_Decouple_shallow(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet_Decouple_shallow, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        # self.block_nine_diff = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        # self.block_nine_dist = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv_diff = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv_dist = nn.Conv3d(n_filters, n_classes, 1, padding=0)


        self.cls_indicator = torch.Tensor(n_classes).cuda()

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def encoder(self, input):
        # print(input.size())
        x1 = self.block_one(input)
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

    def decoder(self, features):
        x1, x2, x3, x4, x5 = features
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
        # x9_diff = self.block_nine_diff(x8_up)
        # x9_dist = self.block_nine_dist(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
            # x9_diff = self.dropout(x9_diff)
            # x9_dist = self.dropout(x9_dist)

        return x9

    def forward(self, image, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        features = self.encoder(image)
        x9 = self.decoder(features)

        out_org = self.out_conv(x9)
        out_diff = self.out_conv_diff(x9)
        out_dist = self.out_conv_dist(x9)

        if turnoff_drop:
            self.has_dropout = has_dropout

        return out_org, out_diff, out_dist







class VNet_3teachers(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet_3teachers, self).__init__()
        self.has_dropout = has_dropout


        self.encoder = Encoder(n_channels, n_filters, normalization, has_dropout)
        self.decoder_1 = Decoder(n_classes, n_filters, normalization, has_dropout)
        self.decoder_2 = Decoder(n_classes, n_filters, normalization, has_dropout)
        self.decoder_3 = Decoder(n_classes, n_filters, normalization, has_dropout)
        self.decoder_4 = Decoder(n_classes, n_filters, normalization, has_dropout)

        # self.cls_indicator = torch.Tensor(n_classes).cuda()





    def forward(self, image):
        # if turnoff_drop:
        #     has_dropout = self.has_dropout
        #     self.has_dropout = False

        features = self.encoder(image)
        out_stu = self.decoder_1(features)
        out_major = self.decoder_2(features)
        out_diff = self.decoder_3(features)
        out_dist = self.decoder_4(features)

        # if turnoff_drop:
        #     self.has_dropout = has_dropout

        return out_stu, out_major, out_diff, out_dist
