import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import os
from utils import read_list, read_data, config, softmax


class KneeMRI_light(Dataset):
    def __init__(self, split='train', repeat=None, transform=None, unlabeled=False):
        self.ids_list = read_list(split)
        self.repeat = repeat
        if self.repeat is None:
            self.repeat = len(self.ids_list)
        print('total {} datas'.format(self.repeat))
        self.transform = transform
        self.unlabeled = unlabeled
        self.num_cls = config.num_cls
        self._weight = None

    def __len__(self):
        return self.repeat

    def _get_data(self, data_id):
        # [160, 384, 384]
        image, label = read_data(data_id)
        return data_id, image, label

    #@property
    def weight(self):
        if self.unlabeled:
            raise ValueError
        if self._weight is not None:
            return self._weight
        weight = np.zeros(self.num_cls)
        for data_id in self.ids_list:
            _, _, label = self._get_data(data_id)
            label = label.reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            weight += tmp

        weight = weight.astype(np.float32)
        # weight = weight / np.sum(weight)
        self._weight = np.power(np.amax(weight) / weight, 1/3)
        return self._weight, weight

    def weight_para(self):
        if self.unlabeled:
            raise ValueError
        if self._weight is not None:
            return self._weight
        weight = np.zeros(self.num_cls)
        for data_id in self.ids_list:
            _, _, label = self._get_data(data_id)
            label = label.reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            weight += tmp

        weight = weight.astype(np.float32)
        # weight = weight / np.sum(weight)
        self._weight = np.power(np.amax(weight) / weight, 1/3)
        return self._weight, weight

    def init_crest_weight(self):
        if self.unlabeled:
            raise ValueError
        if self._weight is not None:
            return self._weight
        weight = np.zeros(self.num_cls)
        for data_id in self.ids_list:
            _, _, label = self._get_data(data_id)
            label = label.reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            weight += tmp

        weight = weight.astype(np.float32)
        # weight = weight / np.sum(weight)
        self._weight = np.power(np.amax(weight) / weight, 1/3)
        return self._weight, weight

    def weight_simis(self):
        if self.unlabeled:
            raise ValueError
        if self._weight is not None:
            return self._weight
        num_each_class = np.zeros(self.num_cls)
        for data_id in self.ids_list:
            _, _, label = self._get_data(data_id)
            label = label.reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp

        num_each_class = num_each_class[1:].astype(np.float32)
        P = num_each_class / np.sum(num_each_class)
        prob_S = (np.max(P) - P) / self.num_cls
        prob_S = np.insert(prob_S, 0, 0.0)

        self._weight = prob_S + 1.0
        # self._weight = np.power(np.amax(weight) / weight, 1/3)
        return self._weight, num_each_class

    def __getitem__(self, index):
        index = index % len(self.ids_list)
        data_id = self.ids_list[index]
        _, image, label = self._get_data(data_id)
        # print("===",label)
        label = label.astype(np.uint8)

        if self.unlabeled: # <-- for safety
            label = np.zeros_like(image)
            # print(label)
        # print("before",image.min(), image.max())
        # image = (image - image.min()) / (image.max() - image.min())
        # image = image.clip(min=-400)
        # image = (image - image.min()) / (image.max() - image.min()) * 2.0 - 1.0
        image = (image - image.mean()) / (image.std() + 1e-8)
        # print("after",image.min(), image.max())
        # print("ss",image.max())
        # image = image.astype(np.float32)
        # print(image.shape, label.shape)
        sample = {'image': image, 'label': label}

        # print(sample['image'])
        
        if self.transform:
            sample = self.transform(sample)
        return sample


class KneeMRI(KneeMRI_light):
    def __init__(self, split='train', repeat=None, transform=None, unlabeled=False):
        super().__init__(split=split, repeat=repeat, transform=transform, unlabeled=unlabeled)
        self.data_list = {}
        for data_id in tqdm(self.ids_list): # <-- load data to memory
            image, label = read_data(data_id)
            self.data_list[data_id] = (image, label)

    def _get_data(self, data_id):
        image, label = self.data_list[data_id]
        return data_id, image, label




class Synapse(Dataset):
    def __init__(self, split='train', repeat=None, transform=None, unlabeled=False, is_val=False):
        self.ids_list = read_list(split)
        self.repeat = repeat
        if self.repeat is None:
            self.repeat = len(self.ids_list)
        print('total {} datas'.format(self.repeat))
        self.transform = transform
        self.unlabeled = unlabeled
        self.num_cls = config.num_cls
        self._weight = None
        self.is_val = is_val
        if self.is_val:
            self.data_list = {}
            for data_id in tqdm(self.ids_list): # <-- load data to memory
                image, label = read_data(data_id)
                self.data_list[data_id] = (image, label)


    def __len__(self):
        return self.repeat

    def _get_data(self, data_id):
        # [160, 384, 384]
        if self.is_val:
            image, label = self.data_list[data_id]
        else:
            image, label = read_data(data_id)
        return data_id, image, label

    def __getitem__(self, index):
        index = index % len(self.ids_list)
        data_id = self.ids_list[index]
        _, image, label = self._get_data(data_id)
        if self.unlabeled: # <-- for safety
            label[:] = 0
        # print("before",image.min(), image.max())
        # image = (image - image.min()) / (image.max() - image.min())
        image = image.clip(min=-75, max=275)
        image = (image - image.min()) / (image.max() - image.min())
        # image = (image - image.mean()) / (image.std() + 1e-8)
        # print("after",image.min(), image.max())
        # print("ss",image.max())
        image = image.astype(np.float32)

        # print(image.shape, label.shape)
        sample = {'image': image, 'label': label}

        # print(sample['image'])

        if self.transform:
            # if not self.unlabeled and not self.is_val:
            #     sample = self.transform(sample, weights=self.transform.weights)
            # else:
            sample = self.transform(sample)

        return sample
