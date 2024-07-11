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
import albumentations as A

import copy
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose, AbstractTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform, Rot90Transform
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, SegChannelSelectionTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform


from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform, BrightnessTransform

from batchgenerators.augmentations.spatial_transformations import augment_spatial, augment_mirroring
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_noise, augment_gaussian_blur

from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, rotate_coords_2d, rotate_coords_3d, scale_coords
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug


import numpy as np
import albumentations as A
import cv2


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


def augment_scale(data, seg, patch_size, patch_center_dist_from_border=30,
                  scale=(0.6, 1.0), border_mode_data='constant', border_cval_data=0, order_data=3,
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




class TrivialAugment(Compose):
    def __init__(self, transforms, sample_num=1):
        super(TrivialAugment, self).__init__(transforms)
        self.transforms = transforms
        self.sample_num = sample_num

    def pre_post_aug(self, list):
        self.list = list



    def __call__(self, data_dict):

        tr_transforms = random.sample(self.transforms, k=self.sample_num)
        # print(tr_transforms)
        list = copy.deepcopy(self.list)
        if tr_transforms is not None:
            for i in range(len(tr_transforms)):
                list.insert(3, tr_transforms[i])
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



class ScaleTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1.0, scale=(0.6, 1.0)):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample
        self.scale = scale


    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        label = data_dict[self.label_key]
        for b in range(data.shape[0]):
            if random.random() < self.p_per_sample:
                data, label = augment_scale(data, label, patch_size=data.shape[2:5], scale=self.scale)
        data_dict[self.data_key] = data
        data_dict[self.label_key] = label
        return data_dict

class RotationTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1.0):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        label = data_dict[self.label_key]
        for b in range(data.shape[0]):
            if random.random() < self.p_per_sample:
                data, label = augment_rotation(data, label, patch_size=data.shape[2:5])
        data_dict[self.data_key] = data
        data_dict[self.label_key] = label
        return data_dict


class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        ret_dict = {}
        resize_shape=(self.output_size[0],
                      self.output_size[1],
                      self.output_size[2])
        for key in sample.keys():
            item = sample[key]
            item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
            # print(item.shape)

            item = F.interpolate(item, size=resize_shape, mode="nearest")
            item = item.squeeze().numpy()
            ret_dict[key] = item

        return ret_dict




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
        # resize_shape=(self.output_size[0]+self.output_size[0]//8,
        #               self.output_size[1]+self.output_size[1]//8,
        #               self.output_size[2]+self.output_size[2]//8)
        # print(resize_shape)x


        for key in data_dict.keys():
            item = data_dict[key]
            item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
            # print(item.shape)
            # if key == 'label':
            #     item = F.interpolate(item, size=resize_shape, mode="nearest")
            # print("img",item.shape)
            # else:
            #     item = F.interpolate(item, size=resize_shape,mode='trilinear', align_corners=False)
            # print("lbl",item.shape)
            item = item.squeeze().numpy()
            if padding_flag:
                item = np.pad(item, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            if w1 is None:
                (w, h, d) = item.shape
                w1 = np.random.randint(0, w - self.output_size[0])
                h1 = np.random.randint(0, h - self.output_size[1])
                d1 = np.random.randint(0, d - self.output_size[2])

            item = item[w1:w1+self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            # print(item.shape)
            ret_dict[key] = item[np.newaxis, np.newaxis, ...]

        return ret_dict


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        padding_flag = image.shape[0] <= self.output_size[0] or \
                       image.shape[1] <= self.output_size[1] or \
                       image.shape[2] <= self.output_size[2]

        # pad the sample if necessary
        if padding_flag:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)

        w1, h1, d1 = None, None, None
        ret_dict = {}
        # resize_shape=(self.output_size[0]+2,
        #               self.output_size[1]+2,
        #               self.output_size[2]+2)


        for key in sample.keys():
            item = sample[key]
            # item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
            # print(item.shape)
            # if key == 'image':
            #     item = F.interpolate(item, size=resize_shape,mode='trilinear', align_corners=False)
            # else:
            #     item = F.interpolate(item, size=resize_shape, mode="nearest")
            # print(item.max())
            # item = item.squeeze().numpy()
            if padding_flag:
                item = np.pad(item, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            if w1 is None:
                (w, h, d) = item.shape
                w1 = int(round((w - self.output_size[0]) / 2.))
                h1 = int(round((h - self.output_size[1]) / 2.))
                d1 = int(round((d - self.output_size[2]) / 2.))

            item = item[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            ret_dict[key] = item

        return ret_dict



class RandomFlip(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1.0):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        label = data_dict[self.label_key]
        for b in range(data.shape[0]):
            if random.random() < self.p_per_sample:
                axis = np.random.randint(0, 2)
                data[b] = np.flip(data[b], axis=axis).copy()
                label[b] = np.flip(label[b], axis=axis).copy()
                # data, label = augment_rotation(data, label, patch_size=data.shape[2:5])
        data_dict[self.data_key] = data
        data_dict[self.label_key] = label
        return data_dict


class ElasticTransform(AbstractTransform):
    def __init__(self, data_key="data", seg_key="seg", p_per_sample=1.0, z_last=False):
        self.opt = A.augmentations.geometric.transforms.ElasticTransform(value=0, mask_value=0)
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.z_last = z_last

    def __call__(self, **data_dict):
        feature, target = data_dict[self.data_key], data_dict[self.seg_key]
        params = self.opt.get_params()

        if self.z_last:
            feature = feature.transpose((0, 1, 3, 4, 2))
            target = target.transpose((0, 1, 3, 4, 2))

        for batch_index in range(feature.shape[0]):
            if random.random() < self.p_per_sample:
                for channel_index in range(feature.shape[1]):
                    for z_index in range(feature.shape[2]):
                        feature[batch_index, channel_index, z_index] = self.opt.apply(
                            feature[batch_index, channel_index, z_index], **params
                        )
                        if channel_index == 0:
                            target[batch_index, channel_index, z_index] = self.opt.apply_to_mask(
                                target[batch_index, channel_index, z_index], **params
                            )

        if self.z_last:
            feature = feature.transpose((0, 1, 4, 2, 3))
            target = target.transpose((0, 1, 4, 2, 3))

        data_dict[self.data_key], data_dict[self.seg_key] = feature, target
        return data_dict


class GridDistortion(AbstractTransform):
    def __init__(self, data_key="data", seg_key="seg", p_per_sample=1.0, z_last=1):
        self.opt = A.GridDistortion(value=0, mask_value=0)
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.z_last = z_last


    def __call__(self, **data_dict):

        feature, target = data_dict[self.data_key], data_dict[self.seg_key]
        transform = A.GridDistortion(distort_limit=(0.2, 0.4),border_mode=cv2.BORDER_CONSTANT, p=1)
        # print(feature.shape)

        if self.z_last:
            feature = feature.transpose((0, 1, 3, 4, 2))
            target = target.transpose((0, 1, 3, 4, 2))


        for batch_index in range(feature.shape[0]):
            if random.random() < self.p_per_sample:
                for channel_index in range(feature.shape[1]):
                    for z_index in range(feature.shape[2]):
                        feature[batch_index, channel_index, z_index] = transform(image=feature[batch_index, channel_index, z_index]
                                                                                 )['image']
                        if channel_index == 0:
                            target[batch_index, channel_index, z_index] = transform(image=target[batch_index, channel_index, z_index]
                                                                                    )['image']

        if self.z_last:
            feature = feature.transpose((0, 1, 4, 2, 3))
            target = target.transpose((0, 1, 4, 2, 3))

        data_dict[self.data_key], data_dict[self.seg_key] = feature, target
        return data_dict


class SharpenTransform(AbstractTransform):
    def __init__(self, data_key="data", seg_key="seg", p_per_sample=1.0, z_last=False):
        self.opt = A.augmentations.transforms.Sharpen(alpha=(0.2, 0.8), p=1)
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.z_last = z_last

    def __call__(self, **data_dict):
        feature, target = data_dict[self.data_key], data_dict[self.seg_key]
        params = self.opt.get_params()
        # save = sitk.GetImageFromArray(feature[0][0])
        # sitk.WriteImage(save, '../before_CLAHE.nii.gz')

        if self.z_last:
            feature = feature.transpose((0, 1, 3, 4, 2))
            target = target.transpose((0, 1, 3, 4, 2))

        for batch_index in range(feature.shape[0]):
            if random.random() < self.p_per_sample:
                for channel_index in range(feature.shape[1]):
                    for z_index in range(feature.shape[2]):
                        feature[batch_index, channel_index, z_index] = self.opt.apply(
                            feature[batch_index, channel_index, z_index], **params
                        )

        if self.z_last:
            feature = feature.transpose((0, 1, 4, 2, 3))
            target = target.transpose((0, 1, 4, 2, 3))
        # save = sitk.GetImageFromArray(feature[0][0])
        # sitk.WriteImage(save, '../CLAHE.nii.gz')
        data_dict[self.data_key], data_dict[self.seg_key] = feature, target
        return data_dict


def get_StrongAug(patch_size, sample_num):
    tr_transforms = []
    tr_transforms_select = []
    tr_transforms.append(RandomCrop(patch_size))
    tr_transforms.append(RenameTransform('image', 'data', True))
    tr_transforms.append(RenameTransform('label', 'seg', True))
    p_per_sample=0.3

    # ========== Spatial-level Transforms =================
    tr_transforms_select.append(RotationTransform(p_per_sample=p_per_sample))
    tr_transforms_select.append(ScaleTransform(scale=(0.5, 1.0) , p_per_sample=p_per_sample))
    # tr_transforms_select.append(RandomFlip(p_per_sample=p_per_sample))

    # ========== Pixel-level Transforms =================
    tr_transforms_select.append(GaussianBlurTransform((0.7, 1.3), p_per_sample=p_per_sample))
    tr_transforms_select.append(BrightnessMultiplicativeTransform(p_per_sample=p_per_sample))
    tr_transforms_select.append(ContrastAugmentationTransform(contrast_range=(0.5, 1.5), p_per_sample=p_per_sample))
    tr_transforms_select.append(GammaTransform(invert_image=False, per_channel=True, retain_stats=True, p_per_sample=p_per_sample))  # inverted gamma
    # tr_transforms_select.append(SimulateLowResolutionTransform(p_per_sample=p_per_sample))
    # tr_transforms_select.append(CLAHETransform(p_per_sample=p_per_sample))
    # tr_transforms_select.append(SharpenTransform(p_per_sample=p_per_sample))



    tr_transforms.append(RemoveLabelTransform(-1, 0))

    tr_transforms.append(RenameTransform('seg', 'label', True))
    tr_transforms.append(RenameTransform('data', 'image', True))
    tr_transforms.append(NumpyToTensor(['image', 'label'], 'float'))
    trivialAug = TrivialAugment(tr_transforms_select, sample_num)
    trivialAug.pre_post_aug(tr_transforms)
    return trivialAug

