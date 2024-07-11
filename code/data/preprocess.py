import os
import glob
import numpy as np
from tqdm import tqdm

from utils import read_list, read_nifti, config
import torch
import torch.nn.functional as F
import SimpleITK as sitk
base_dir = config.base_dir


eval_ids = ['10765046_L',
            '10765046_R',
            '10922176_L',
            '10922176_R',
            '11169950_L',
            '11169950_R',
            '11274563_L',
            '11274563_R',
            '11278334_L',
            '11278334_R',
            '11289583_L',
            '11289583_R',
            '11291463_L',
            '11291463_R',
            '11303592_L',
            '11303592_R',
            # old
            '11200756_L',
            '11200756_R',
            '11278166_L',
            '11278166_R',
            '11286881_L',
            '11286881_R',
            '11292583_L',
            '11292583_R',
            '11296280_L',
            '11296280_R',
            '11306795_L',
            '11306795_R',
            ]


test_ids = ['11298887_L',
            '11298887_R',
            '11301863_L',
            '11301863_R',
            '11212400_L',
            '11212400_R',
            '11286134_L',
            '11286134_R',
            '11289582_L',
            '11289582_R',
            '11294188_L',
            '11294188_R',
            '11301379_L',
            '11301379_R',
            '11303741_L',
            '11303741_R',
            # new
            '11112693_L',
            '11112693_R',
            '11283204_L',
            '11283204_R',
            '11288236_L',
            '11288236_R',
            '11288949_L',
            '11288949_R',
            '11292540_L',
            '11292540_R',
            '11293841_L',
            '11293841_R',
            '11301674_L',
            '11301674_R',
            ]

val_test_ids = eval_ids + test_ids



def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')


def process_npy():
    for tag in ['Tr']:
        img_ids = []
        for path in tqdm(glob.glob(os.path.join(base_dir, f'images{tag}', '*.nii.gz'))):
            # print(path)
            img_id = path.split('/')[-1].split('.')[0]
            # print(img_id)

            img_ids.append(img_id)
            label_id= img_id[:-4]

            if label_id not in val_test_ids:
                continue

            print(label_id)

            image_path = os.path.join(base_dir, f'images{tag}', f'{img_id}.nii.gz')
            label_path =os.path.join(base_dir, f'labels{tag}', f'{label_id}.nii.gz')

            image = read_nifti(image_path)
            image = image.astype(np.float32)

            # print(image.shape)

            d, w, h = image.shape

            w_s = w // 2 - w // 4
            w_e = w // 2 + w // 4

            h_s = h // 2 - h // 4
            h_e = h // 2 + h // 4

            image = image[:,  w_s:w_e, h_s:h_e]

            if os.path.exists(label_path):
                print(label_path)
                label = read_nifti(label_path)
                label = label.astype(np.int8)
                label = label[:, h_s:h_e, w_s:w_e]

                print("label shape",label.shape)


            print(image.max(), image.min())


            if not os.path.exists(os.path.join(config.save_dir, 'npy')):
                os.makedirs(os.path.join(config.save_dir, 'npy'))

            if not os.path.exists(os.path.join(config.save_dir, 'processed')):
                os.makedirs(os.path.join(config.save_dir, 'processed'))


            np.save(
                os.path.join(config.save_dir, 'npy', f'{img_id[:-4]}_image.npy'),
                image
            )
            img_itk_new = sitk.GetImageFromArray(image)
            sitk.WriteImage(img_itk_new, os.path.join(config.save_dir, 'processed', f'{img_id[:-4]}_image.nii.gz'))

            if os.path.exists(label_path):
                # print(label_path)
                np.save(
                    os.path.join(config.save_dir, 'npy', f'{label_id}_label.npy'),
                    label
                )
                lbl_itk_new = sitk.GetImageFromArray(label)
                sitk.WriteImage(lbl_itk_new, os.path.join(config.save_dir, 'processed', f'{label_id}_label.nii.gz'))


def process_npy_test_newdata():
    img_ids = []
    for path in tqdm(glob.glob(os.path.join(base_dir, f'imagesTs', '*.nii.gz'))):
        # print(path)
        img_id = path.split('/')[-1].split('.')[0]
        # print(img_id)


        pure_id= img_id[:-4]

        img_ids.append(pure_id)



        image_path = os.path.join(base_dir, f'imagesTs', f'{img_id}.nii.gz')

        image = read_nifti(image_path)
        image = image.astype(np.float32)

        d, w, h = image.shape

        w_s = w // 2 - w // 4
        w_e = w // 2 + w // 4

        h_s = h // 2 - h // 4
        h_e = h // 2 + h // 4

        image = image[:,  w_s:w_e, h_s:h_e]


        if not os.path.exists(os.path.join(config.save_dir, 'npy')):
            os.makedirs(os.path.join(config.save_dir, 'npy'))

        if not os.path.exists(os.path.join(config.save_dir, 'processed')):
            os.makedirs(os.path.join(config.save_dir, 'processed'))

        np.save(
            os.path.join(config.save_dir, 'npy', f'{img_id[:-4]}_image.npy'),
            image
        )
        img_itk_new = sitk.GetImageFromArray(image)
        sitk.WriteImage(img_itk_new, os.path.join(config.save_dir, 'processed', f'{img_id[:-4]}_image.nii.gz'))

    write_txt(
        img_ids,
        os.path.join(config.save_dir, 'splits_new/test_new.txt')
    )


def process_split_fully(train_val_ratio=7/8):
    img_ids = []
    label_ids = []
    for path in tqdm(glob.glob(os.path.join(config.save_dir, 'npy', '*_image.npy'))):
        # print(path)
        img_id = path.split('/')[-1].split('.')[0][:-6]
        # print(img_id)
        img_ids.append(img_id)

    for path in tqdm(glob.glob(os.path.join(config.save_dir, 'npy', '*_label.npy'))):
        # print(path)
        label_id = path.split('/')[-1].split('.')[0][:-6]
        # print(img_id)
        label_ids.append(label_id)



    print(len(label_ids))





    train_label_ids = np.setdiff1d(label_ids, val_test_ids)
    print(len(train_label_ids))
    print(len(eval_ids))
    print(len(test_ids))
    # split_idx = int(len(img_ids) * train_ratio)
    train_ids = np.setdiff1d(img_ids, val_test_ids)

    # train_val_ids = np.random.permutation(train_val_ids)

    # print(train_val_ids)
    # test_ids = test_ids

    # split_idx = int(len(train_val_ids) * train_val_ratio)
    # train_ids = sorted(train_val_ids[:split_idx])
    # eval_ids = sorted(train_val_ids[split_idx:])

    if not os.path.exists(os.path.join(config.save_dir, 'splits_new')):
        os.makedirs(os.path.join(config.save_dir, 'splits_new'))

    write_txt(
        train_label_ids,
        os.path.join(config.save_dir, 'splits_new/train_l.txt')
    )

    write_txt(
        train_ids,
        os.path.join(config.save_dir, 'splits_new/train.txt')
    )
    write_txt(
        eval_ids,
        os.path.join(config.save_dir, 'splits_new/eval.txt')
    )

    write_txt(
        test_ids,
        os.path.join(config.save_dir, 'splits_new/test.txt')
    )




def process_split_test():
    img_ids = []
    # label_ids = []
    for path in tqdm(glob.glob(os.path.join(config.save_dir, 'npy', '*_image.npy'))):
        # print(path)
        img_id = path.split('/')[-1].split('.')[0][:-6]
        # print(img_id)
        img_ids.append(img_id)


    if not os.path.exists(os.path.join(config.save_dir, 'splits_new')):
        os.makedirs(os.path.join(config.save_dir, 'splits_new'))

    write_txt(
        img_ids,
        os.path.join(config.save_dir, 'splits_new/test.txt')
    )



def process_split_semi(split='train', labeled_ratio=0.4):
    ids_list = read_list(split)
    # ids_list = np.random.permutation(ids_list)
    labeled_ids = read_list('train_l')

    # split_idx = int(len(ids_list) * labeled_ratio)
    # labeled_ids = sorted(ids_list[:split_idx])
    unlabeled_ids = np.setdiff1d(ids_list, labeled_ids)
    
    write_txt(
        labeled_ids,
        os.path.join(config.save_dir, 'splits_new/labeled.txt')
    )
    write_txt(
        unlabeled_ids,
        os.path.join(config.save_dir, 'splits_new/unlabeled.txt')
    )


if __name__ == '__main__':
    # process_npy()
    process_npy_test_newdata()
    # process_split_test()
    # process_split_fully()
    # process_split_semi()
