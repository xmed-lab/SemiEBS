import torch
import numpy as np
from skimage import transform
import torch.nn.functional as F
from utils import config

class CenterCrop_brain(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        # print("shape",image[0].shape)
        padding_flag = image[0].shape[0] <= self.output_size[0] or \
                       image[0].shape[1] <= self.output_size[1] or \
                       image[0].shape[2] <= self.output_size[2]

        # pad the sample if necessary
        if padding_flag:
            pw = max((self.output_size[0] - image[0].shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image[0].shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image[0].shape[2]) // 2 + 3, 0)

        w1, h1, d1 = None, None, None
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if key == 'image':
                # print("====",item.shape)
                item_new_list = []
                for m in range(item.shape[0]):
                    if padding_flag:
                        item_new = np.pad(item[m], [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                    else: item_new = item[m]
                    if w1 is None:
                        (w, h, d) = item_new.shape
                        w1 = int(round((w - self.output_size[0]) / 2.))
                        h1 = int(round((h - self.output_size[1]) / 2.))
                        d1 = int(round((d - self.output_size[2]) / 2.))
                    item_new_list.append(item_new[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]])
                ret_dict[key] = np.array(item_new_list)
            else:
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
        # resize_shape=(self.output_size[0], self.output_size[1]+50, self.output_size[2]+50)

        if w1 is None:
            (w, h, d) = image.shape
            w1 = int(round((w - self.output_size[0]) / 2.))
            h1 = int(round((h - self.output_size[1]) / 2.))
            d1 = int(round((d - self.output_size[2]) / 2.))
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

            item = item[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            ret_dict[key] = item

        return ret_dict


class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
            # print(item.shape)
            if key == 'image':
                item = F.interpolate(item, size=(self.output_size[0], self.output_size[1], self.output_size[2]),mode='trilinear', align_corners=False)
            else:
                item = F.interpolate(item, size=(self.output_size[0], self.output_size[1], self.output_size[2]), mode="nearest")
            item = item.squeeze().numpy()
            ret_dict[key] = item

        return ret_dict





class RandomCrop(object):
    '''
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    '''
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
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


        for key in sample.keys():
            item = sample[key]
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
            ret_dict[key] = item

        return ret_dict



class RandomCropLA(object):
    '''
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    '''
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        print(image.shape)
        padding_flag = image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]
        # pad the sample if necessary
        if padding_flag:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)

        w1, h1, d1 = None, None, None
        ret_dict = {}
        # print(image.shape)
        resize_shape=(image.shape[0]+20, self.output_size[1]+20, self.output_size[2])
        if w1 is None:
            (w, h, d) = resize_shape
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])
            d1 = np.random.randint(0, d - self.output_size[2])

        for key in sample.keys():
            item = sample[key]
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
            ret_dict[key] = item

        return ret_dict


class RandomCrop_cz_brain(object):
    def __init__(self, output_size, span, p=1):
        self.p = p
        self.span = span
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        padding_flag = image[0].shape[0] <= self.output_size[0] or \
                       image[0].shape[1] <= self.output_size[1] or \
                       image[0].shape[2] <= self.output_size[2]

        # pad the sample if necessary
        if padding_flag:
            pw = max((self.output_size[0] - image[0].shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image[0].shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image[0].shape[2]) // 2 + 3, 0)
        
        w1, h1, d1 = None, None, None
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if key == 'image':
                item_new_list = []
                for m in range(item.shape[0]):
                    if padding_flag:
                        item_new = np.pad(item[m], [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                    else: item_new = item[m]
                    # print(item_new.shape)
                    if w1 is None:
                        (w, h, d) = item_new.shape
                        if np.random.randint(self.p + 1) >= 1:
                            w1 = np.random.randint(self.span, w - self.output_size[0] - self.span)
                        else:
                            w1 = np.random.randint(0, w - self.output_size[0])
                        h1 = np.random.randint(0, h - self.output_size[1])
                        d1 = np.random.randint(0, d - self.output_size[2])
                    item_new_list.append(item_new[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]])
                ret_dict[key] = np.array(item_new_list)
            else:
                if padding_flag:
                    item = np.pad(item, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                if w1 is None:
                    (w, h, d) = item.shape
                    if np.random.randint(self.p + 1) >= 1:
                        w1 = np.random.randint(self.span, w - self.output_size[0] - self.span)
                    else:
                        w1 = np.random.randint(0, w - self.output_size[0])
                    h1 = np.random.randint(0, h - self.output_size[1])
                    d1 = np.random.randint(0, d - self.output_size[2])
                item = item[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
                ret_dict[key] = item
        return ret_dict

class RandomCrop_cz(object):
    def __init__(self, output_size, span, p=1):
        self.p = p
        self.span = span
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        # image_new = []
        image = transform.resize(image, (image.shape[0], self.output_size[1]+10, self.output_size[2]+10))
        # print(image.shape)
        # for d in range(image.shape[0]):
        #     image_new.append(np.resize(image, (256, 256)))
        # image = np.array(image_new)
        # print(image.shape)

        padding_flag = image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]

        # pad the sample if necessary
        if padding_flag:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)

        w1, h1, d1 = None, None, None
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            item = torch.FloatTensor(item).unsqueeze(0).unsqueeze(0)
            # print(item.shape)
            if key == 'image':
                item = F.interpolate(item, size=(self.output_size[0]+10, self.output_size[1]+20, self.output_size[2]+20),mode='trilinear', align_corners=False)
                # print(item.shape)
            else:
                item = F.interpolate(item, size=(self.output_size[0]+10, self.output_size[1]+20, self.output_size[2]+20), mode="nearest")
                # print(item.max())
            item = item.squeeze().numpy()
            # print(item.shape)

            if padding_flag:
                item = np.pad(item, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if w1 is None:
                (w, h, d) = item.shape
                if np.random.randint(self.p + 1) >= 1:
                    w1 = np.random.randint(self.span, w - self.output_size[0] - self.span)
                else:
                    w1 = np.random.randint(0, w - self.output_size[0])
                h1 = np.random.randint(0, h - self.output_size[1])
                d1 = np.random.randint(0, d - self.output_size[2])
            item = item[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            ret_dict[key] = item

        return ret_dict

class RandomRotFlip(object):
    '''
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0, 1) < self.p:
            k = np.random.randint(0, 4)
            axis = np.random.randint(0, 2)
            ret_dict = {}
            for key in sample.keys():
                item = sample[key]
                item = np.rot90(item, k)
                item = np.flip(item, axis=axis).copy()
                ret_dict[key] = item
            return ret_dict
        else:
            return sample
        



class RandomNoise(object):
    def __init__(self,sigma=0.01):
        self.sigma = sigma
        self.mu = 0

    def __call__(self, sample):
        ret_dict = {}
        for key in sample.keys():
            if key == 'image':
                image = sample[key]
                noise = np.clip(
                    self.sigma * np.random.randn(*image.shape),
                    -2 * self.sigma, 
                     2 * self.sigma
                )
                noise = noise + self.mu
                image = image + noise
                ret_dict[key] = image
            else:
                ret_dict[key] = sample[key]
        
        return ret_dict

class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = np.flip(img,1).copy()
        return img

    def __call__(self, sample):
        prob = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
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

    def __call__(self, sample):
        prob = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            self._flip(item, prob)
            ret_dict[key] = item
        return ret_dict


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        ret_dict = {}
        for key in sample.keys():
            item = sample[key]
            if key == 'image':
                # print(item.max())
                ret_dict[key] = torch.from_numpy(item).unsqueeze(0).float()
            elif key == 'label':
                # item[item>config.num_cls-1]=0
                ret_dict[key] = torch.from_numpy(item).long()
            else:
                raise ValueError(key)
        # print(ret_dict['image'].shape)
        
        return ret_dict
