# -*- coding: utf-8 -*-
# @Time    : 2023/3/31 17:57
# @Author  : Haonan Wang
# @File    : AdaptiveAug.py
# @Software: PyCharm
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import random
import torch
import torch.nn.functional as F

import copy
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose, AbstractTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, SegChannelSelectionTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform


from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_brightness_additive, \
    augment_brightness_multiplicative, augment_gamma
from batchgenerators.augmentations.spatial_transformations import augment_spatial, augment_mirroring
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_noise, augment_gaussian_blur

from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, rotate_coords_2d, rotate_coords_3d, scale_coords
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug


import numpy as np
import albumentations as A
import cv2

from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import interpn
from medpy.metric import binary




def augment_rotation(data, seg, patch_size, patch_center_dist_from_border=30,
                     angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                     border_mode_data='constant', border_cval_data=0, order_data=3,
                     border_mode_seg='constant', border_cval_seg=0, order_seg=1, random_crop=False):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        a_x = np.random.uniform(angle_x[0], angle_x[1])
        if dim == 3:
            a_y = 0
            a_z = 0
            coords = rotate_coords_3d(coords, a_x, a_y, a_z)
        else:
            coords = rotate_coords_2d(coords, a_x)

        # now find a nice center location
        for d in range(dim):
            if random_crop:
                ctr = np.random.uniform(patch_center_dist_from_border[d],
                                        data.shape[d + 2] - patch_center_dist_from_border[d])
            else:
                ctr = data.shape[d + 2] / 2. - 0.5
            coords[d] += ctr
        for channel_id in range(data.shape[1]):
            data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                 border_mode_data, cval=border_cval_data)
        if seg is not None:
            for channel_id in range(seg.shape[1]):
                seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                    border_mode_seg, cval=border_cval_seg,
                                                                    is_seg=True)
    return data_result, seg_result


def augment_elastic(data, seg, patch_size, patch_center_dist_from_border=30,
                    alpha=(0., 900.), sigma=(9., 13.),
                    border_mode_data='nearest', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=False):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)

        a = np.random.uniform(alpha[0], alpha[1])
        s = np.random.uniform(sigma[0], sigma[1])
        coords = elastic_deform_coordinates(coords, a, s)

        # now find a nice center location
        for d in range(dim):
            if random_crop:
                ctr = np.random.uniform(patch_center_dist_from_border[d],
                                        data.shape[d + 2] - patch_center_dist_from_border[d])
            else:
                ctr = data.shape[d + 2] / 2. - 0.5
            coords[d] += ctr
        for channel_id in range(data.shape[1]):
            data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                 border_mode_data, cval=border_cval_data)
        if seg is not None:
            for channel_id in range(seg.shape[1]):
                seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                    border_mode_seg, cval=border_cval_seg,
                                                                    is_seg=True)

    return data_result, seg_result


def augment_scale(data, seg, patch_size, patch_center_dist_from_border=30,
                  scale=(0.7, 1.4), border_mode_data='constant', border_cval_data=0, order_data=3,
                  border_mode_seg='constant', border_cval_seg=0, order_seg=1, random_crop=False,
                  independent_scale_for_each_axis=False, p_independent_scale_per_axis: int = 1):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)

        if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
            sc = []
            for _ in range(dim):
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc.append(np.random.uniform(scale[0], 1))
                else:
                    sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
        else:
            if np.random.random() < 0.5 and scale[0] < 1:
                sc = np.random.uniform(scale[0], 1)
            else:
                sc = np.random.uniform(max(scale[0], 1), scale[1])
        coords = scale_coords(coords, sc)

        # print("---------------------------------111")

        # now find a nice center location
        for d in range(dim):
            ctr = data.shape[d + 2] / 2. - 0.5
            coords[d] += ctr


        for channel_id in range(data.shape[1]):
            data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                 border_mode_data, cval=border_cval_data)
        # print("---------------------------------222")
        if seg is not None:
            for channel_id in range(seg.shape[1]):
                seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                    border_mode_seg, cval=border_cval_seg,
                                                                    is_seg=True)
        # print("---------------------------------333")
        # print(data_result.shape, seg_result.shape)
    return data_result, seg_result



def transform_list():
    transform_list = ["Contrast",
                      "Brightness",
                      "GAMMA",
                      "GAMMAInvert",
                      "GaussianNoise",
                      "GaussianBlur",
                      "BrightnessMultiplicative",
                      "Scale",
                      "Rotation"]
    return transform_list


class Contrastive2DTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1.0):
        """
        Augments the brightness of data. Additive brightness is sampled from Gaussian distribution with mu and sigma
        :param mu: mean of the Gaussian distribution to sample the added brightness from
        :param sigma: standard deviation of the Gaussian distribution to sample the added brightness from
        :param per_channel: whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample
        self.transform_names = transform_list()
        # self.transform_names = ["Scale",] #"Elastic", "Rotation"]
        self.trans_num = len(self.transform_names)
        # self.weights = weights

    @staticmethod
    def getRangeImageDepth(label):
        # print(label.shape)
        # print(label.max())
        z = np.any(label, axis=(1,2)) # z.shape:(depth,)
        # print(np.where(z)[0])
        if len(np.where(z)[0]) >0:
            startposition,endposition = np.where(z)[0][[0,-1]]
        else:
            startposition = endposition = 0
        return startposition, endposition

    def apply_transform(self, image, label, transform_name):
        if transform_name == "Contrast":
            return augment_contrast(image, contrast_range=(0.5, 1.5)), label
        elif transform_name == "Brightness":
            return augment_brightness_additive(image, 0.0, 0.1), label
        elif transform_name == "BrightnessMultiplicative":
            return augment_brightness_multiplicative(image, multiplier_range=(0.5, 1.5)), label
        elif transform_name == "GAMMA":
            return augment_gamma(image, gamma_range=(0.5, 1.5), invert_image=False), label
        elif transform_name == "GAMMAInvert":
            return augment_gamma(image, gamma_range=(0.5, 1.5), invert_image=True), label
        elif transform_name == "GaussianNoise":
            return augment_gaussian_noise(image), label
        elif transform_name == "GaussianBlur":
            return augment_gaussian_blur(image, (0.5, 1.)), label
        elif transform_name == "Scale":
            image, label = image[np.newaxis], label[np.newaxis, np.newaxis]
            image, label = augment_scale(image, label, patch_size=image.shape[2:5])
            # print(image.shape, label.shape)
            return image.squeeze(0), label.squeeze(0).squeeze(0)
        elif transform_name == "Elastic":
            image, label = image[np.newaxis], label[np.newaxis, np.newaxis]
            image, label =  augment_elastic(image, label, patch_size=image.shape[2:5])
            return image.squeeze(0), label.squeeze(0).squeeze(0)
        elif transform_name == "Rotation":
            image, label = image[np.newaxis], label[np.newaxis, np.newaxis]
            image, label = augment_rotation(image, label, patch_size=image.shape[2:5])
            return image.squeeze(0), label.squeeze(0).squeeze(0)
        else:
            return image, label


    def merge_after(self, data, label, data_aug, label_fg, startIndex,endIndex, half_len):
        data, data_aug = data.transpose((1,0,2,3)), data_aug.transpose((1,0,2,3))
        # print("========", data.shape)
        for i in range(label_fg.shape[0]):
            data=np.insert(data, endIndex+i, data_aug[i], axis=0)
            label=np.insert(label, endIndex+i, label_fg[i], axis=0)
        data, data_aug = data.transpose((1,0,2,3)), data_aug.transpose((1,0,2,3))
        # print("---------", data.shape)
        if endIndex<half_len:
            endIndex=half_len
        elif endIndex>label.shape[0]-half_len:
            endIndex = label.shape[0]-half_len
        data = data[:, endIndex-half_len:endIndex+half_len]
        label = label[endIndex-half_len:endIndex+half_len]
        return data, label

    def merge_intersection(self, data, label, data_aug, label_fg, startIndex,endIndex, half_len):
        data, data_aug = data.transpose((1,0,2,3)), data_aug.transpose((1,0,2,3))
        # print("========", data.shape)
        index = startIndex
        # print(index, endIndex, half_len)
        for i in range(data_aug.shape[0]):
            data=np.insert(data, index+1, data_aug[i], axis=0)
            label=np.insert(label, index+1, label_fg[i], axis=0)
            index+=2
        data, data_aug = data.transpose((1,0,2,3)), data_aug.transpose((1,0,2,3))
        # print("---------", data.shape, label.shape)
        if endIndex<half_len:
            endIndex=half_len
        elif endIndex>label.shape[0]-half_len:
            endIndex = label.shape[0]-half_len
        # elif endIndex>label.shape[0]-half_len:
        #     endIndex = label.shape[0]-half_len
        # print(endIndex-half_len, endIndex+half_len)
        data = data[:, endIndex-half_len:endIndex+half_len]
        label = label[endIndex-half_len:endIndex+half_len]
        # print("=========", data.shape)
        return data, label

    def lesion_copy_intra_case(self, data, label, transform_name):
        half_len = label.shape[0]//2
        startIndex, endIndex = self.getRangeImageDepth(label)
        # print(startIndex, endIndex, endIndex - startIndex)
        data_fg = data[:,startIndex:endIndex+1]
        label_fg = label[startIndex:endIndex+1]
        # print("&&&&&&",data_fg.shape, label_fg.shape)
        data_aug, label_aug = self.apply_transform(data_fg, label_fg, transform_name)
        # print("$$$$$$$$$",transform_name,data_aug.shape, label_aug.shape)
        # if random.random() < 0.5:
        data_res, label_res = self.merge_intersection(data, label, data_aug, label_aug, startIndex,endIndex, half_len)
        # else:
        #     data_res, label_res = self.merge_after(data, label, data_aug, label_aug, startIndex,endIndex, half_len)

        return data_res, label_res




    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        label = data_dict[self.label_key]
        # print("call",self.weights)
        transform_name = random.choices(self.transform_names, k=1)[0]
        # print("====",transform_name)
        label[label<0] = 0
        # print(label.max())
        for b in range(0, data.shape[0]):
            if random.random() < self.p_per_sample:

                #print("++++++++++++++++++++++++++++++++++++++++++++")
                # for c in range(data.shape[1]):
                # save = sitk.GetImageFromArray(data[b][0])
                # sitk.WriteImage(save, '/home/xmli/hnwang/nnFormer/test_aug/'+str(b)+'_org.nii.gz')
                # save = sitk.GetImageFromArray(label[b][0])
                # sitk.WriteImage(save, '/home/xmli/hnwang/nnFormer/test_aug/'+str(b)+'_label.nii.gz')


                # print("before",np.max(data[b]),np.min(data[b]))
                # data, label = data[b], label[b]
                # min1 = np.min(data1)
                # max1 = np.max(data1)
                # data1 = (data1 - min1) / (max1 - min1)
                # min2 = np.min(data2)
                # max2 = np.max(data2)
                # data2 = (data2 - min2) / (max2 - min2)

                # print("after",np.max(data1),np.min(data1))

                # for c in range(data.shape[1]):
                # if random.random() < 0.5:
                #     data1, data2, label1, label2=self.lesion_copy_inter_cases(data1, data2,
                #                                                               label1.squeeze(), label2.squeeze(),
                #                                                               transform_name)
                # else:
                # data, label =self.lesion_copy_intra_case(data, label.squeeze(), transform_name)
                if label[b].max() > 0:
                    # print(data[b].shape, label[b].shape)
                    data[b], label[b] = self.lesion_copy_intra_case(data[b], label[b].squeeze(), transform_name)
                # print("after",np.max(data[b]),np.min(data[b]))

                # for c in range(data.shape[1]):
                # save = sitk.GetImageFromArray(data[b+1][0])
                # sitk.WriteImage(save, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+'_zafter.nii.gz')
                # save = sitk.GetImageFromArray(label[b+1][0])
                # sitk.WriteImage(save, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+'_labelafter.nii.gz')
        data_dict[self.data_key] = data
        data_dict[self.label_key] = label
        # data_dict["trans_name"] = transform_name
        return data_dict




class BodyShapeTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1.0):
        self.data_key = data_key
        self.label_key = label_key
        self.op_axis = [(0,1,2,3),# front and back
                        (0,1,3,2), # left and right
                        (0,3,2,1) # top and bottom
                        ]
        self.p_per_sample = p_per_sample


    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        label = data_dict[self.label_key]
        label[label<0] = 0
        # print("====",data.shape)
        # print("====",label.shape)
        # transform = A.GridDistortion(border_mode=cv2.BORDER_CONSTANT,p=1)
        transform = A.GridDistortion(distort_limit=(0.2, 0.4),border_mode=cv2.BORDER_CONSTANT, p=1)
        # transform = A.GridDistortion(distort_limit=(0.1, 0.4),border_mode=cv2.BORDER_CONSTANT, p=1)
        op_axis = random.choices(self.op_axis, k=1)[0]
        # print("======",op_axis)
        if random.random() < self.p_per_sample:
            for b in range(data.shape[0]):
                data_after = []
                label_after = []
                # min, max = np.min(data[b]), np.max(data[b])
                # for c in range(data.shape[1]):
                # out = sitk.GetImageFromArray(data[b][0])
                # sitk.WriteImage(out, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+'_org.nii.gz')
                # out = sitk.GetImageFromArray(label[b][0])
                # sitk.WriteImage(out, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+'_labelorg.nii.gz')

                data_trans = data[b].transpose(op_axis)
                label_trans = label[b].transpose(op_axis)
                for c in range(data_trans.shape[0]):
                    data_after.append(transform(image=data_trans[c])['image'])
                label_after.append(transform(image=label_trans[0])['image'])
                # print("success")
                data[b] = np.array(data_after).transpose(op_axis)
                label[b] = np.array(label_after).transpose(op_axis)
                # for c in range(data.shape[1]):
                # out = sitk.GetImageFromArray(data[b][0])
                # sitk.WriteImage(out, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+'.nii.gz')
                # out = sitk.GetImageFromArray(label[b][0])
                # sitk.WriteImage(out, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+'_label.nii.gz')
                # data[b] = 2 * (data[b] - min) / (max - min) - 1
                # sys.exit(0)
        data_dict[self.data_key] = data
        data_dict[self.label_key] = label
        # data_dict['trans_name'] = ""
        return data_dict




class TrivialAugment(Compose):

    def pre_post_aug(self, list):
        self.list = list


    def __call__(self, data_dict):

        tr_transforms = random.choices(self.transforms, k=1)[0]
        list = copy.deepcopy(self.list)
        if tr_transforms is not None:
            list.insert(3, tr_transforms)
        # print("1###########################################################################################")
        # for item in list:
        #     print("-----",item)
        # print("1###########################################################################################")

        # print("=======",list)

        for t in list:
            data_dict = t(**data_dict)
        del tr_transforms
        del list

        for key in data_dict.keys():
            if key == "image":
                data_dict[key] = data_dict[key].squeeze(0)
            elif key == "label":
                data_dict[key] = data_dict[key].squeeze(0).squeeze(0)
            # print(data_dict[key].shape).squeeze(1)

        return data_dict



class Compose_new(Compose):


    def __call__(self, data_dict):


        for t in self.transforms:
            data_dict = t(**data_dict)
        # del tr_transforms
        # del list

        for key in data_dict.keys():
            if key == "image":
                data_dict[key] = data_dict[key].squeeze(0)
            elif key == "label":
                data_dict[key] = data_dict[key].squeeze(0).squeeze(0)
            # print(data_dict[key].shape).squeeze(1)

        return data_dict



class RandomCrop(AbstractTransform):
    '''
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    '''
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, **data_dict):
        image = data_dict['image']
        # print(image.shape)
        padding_flag = image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]
        # pad the sample if necessary
        if padding_flag:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)

        w1, h1, d1 = None, None, None
        ret_dict = {}
        # print(image.shape)
        resize_shape=(image.shape[0], self.output_size[1]+50, self.output_size[2]+50)
        # print(resize_shape)
        if w1 is None:
            (w, h, d) = resize_shape
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])
            d1 = np.random.randint(0, d - self.output_size[2])

        for key in data_dict.keys():
            item = data_dict[key]
            item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
            # print(item.shape)
            if key == 'image':
                item = F.interpolate(item, size=resize_shape,mode='trilinear', align_corners=False)
                # print("img",item.shape)
            else:
                item = F.interpolate(item, size=resize_shape, mode="nearest")
                # print("lbl",item.shape)
            item = item.squeeze().numpy()
            if padding_flag:
                item = np.pad(item, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            item = item[w1:w1+self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            # print(item.shape)
            ret_dict[key] = item[np.newaxis, np.newaxis, ...]

        return ret_dict




class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = np.flip(img,1).copy()
        return img

    def __call__(self, **data_dict):
        prob = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        ret_dict = {}
        for key in data_dict.keys():
            item = data_dict[key]
            self._flip(item, prob)
            ret_dict[key] = item
        return ret_dict

class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = np.flip(img, 2).copy()
        return img

    def __call__(self, **data_dict):
        prob = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        ret_dict = {}
        for key in data_dict.keys():
            item = data_dict[key]
            self._flip(item, prob)
            ret_dict[key] = item
        return ret_dict


def get_AdaptiveAugTrain(patch_size):
    # assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

    tr_transforms = []
    tr_transforms_select = []
    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    # if params.get("dummy_2D") is not None and params.get("dummy_2D"):
    #     ignore_axes = (0,)
        # tr_transforms.append(Convert3DTo2DTransform())
    # else:

    # if params.get("selected_data_channels") is not None:
    #     tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    #
    # if params.get("selected_seg_channels") is not None:
    #     tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))
    tr_transforms.append(RandomCrop(patch_size))

    tr_transforms.append(RenameTransform('image', 'data', True))
    tr_transforms.append(RenameTransform('label', 'seg', True))

    # tr_transforms_select.append(None)
    tr_transforms_select.append(BodyShapeTransform(p_per_sample=0.6))
    # tr_transforms_select.append(InterpolationTransform(p_per_sample=1))
    tr_transforms_select.append(Contrastive2DTransform(p_per_sample=0.8))


    # tr_transforms_select.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=1))  # inverted gamma

    # tr_transforms_select.append(GaussianNoiseTransform(p_per_sample=1))
    # tr_transforms_select.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=1, p_per_channel=0.5))
    # tr_transforms_select.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.5))
    # tr_transforms_select.append(ContrastAugmentationTransform(p_per_sample=0.5))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.5,
                                                        ignore_axes=None))

    tr_transforms.append(RandomFlip_LR(0.2))
    tr_transforms.append(RandomFlip_UD(0.2))



    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(RenameTransform('seg', 'label', True))
    tr_transforms.append(RenameTransform('data', 'image', True))
    tr_transforms.append(NumpyToTensor(['image', 'label'], 'float'))


    # print("###########################################################################################")
    # for item in tr_transforms_select:
    #     print("-----",item)
    # print("###########################################################################################")
    # print(tr_transforms)

    trivialAug = TrivialAugment(tr_transforms_select)
    # trivialAug = Compose_new(tr_transforms)
    trivialAug.pre_post_aug(tr_transforms)



    return trivialAug


def get_AdaptiveAug(patch_size):
    # assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

    tr_transforms = []
    tr_transforms_select = []
    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    # if params.get("dummy_2D") is not None and params.get("dummy_2D"):
    #     ignore_axes = (0,)
    # tr_transforms.append(Convert3DTo2DTransform())
    # else:

    # if params.get("selected_data_channels") is not None:
    #     tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    #
    # if params.get("selected_seg_channels") is not None:
    #     tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))
    tr_transforms.append(RandomCrop(patch_size))

    tr_transforms.append(RenameTransform('image', 'data', True))
    tr_transforms.append(RenameTransform('label', 'seg', True))

    # tr_transforms_select.append(None)
    tr_transforms_select.append(BodyShapeTransform(p_per_sample=0.6))
    # tr_transforms_select.append(InterpolationTransform(p_per_sample=1))
    tr_transforms_select.append(Contrastive2DTransform(p_per_sample=0.8))


    # tr_transforms_select.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=1))  # inverted gamma

    # tr_transforms_select.append(GaussianNoiseTransform(p_per_sample=1))
    # tr_transforms_select.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=1, p_per_channel=0.5))
    # tr_transforms_select.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.5))
    # tr_transforms_select.append(ContrastAugmentationTransform(p_per_sample=0.5))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.5,
                                                        ignore_axes=None))

    # tr_transforms.append(RandomFlip_LR(0.2))
    # tr_transforms.append(RandomFlip_UD(0.2))



    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(RenameTransform('seg', 'label', True))
    tr_transforms.append(RenameTransform('data', 'image', True))
    tr_transforms.append(NumpyToTensor(['image', 'label'], 'float'))


    # print("###########################################################################################")
    # for item in tr_transforms_select:
    #     print("-----",item)
    # print("###########################################################################################")
    # print(tr_transforms)

    trivialAug = TrivialAugment(tr_transforms_select)
    # trivialAug = Compose_new(tr_transforms)
    trivialAug.pre_post_aug(tr_transforms)



    return trivialAug

