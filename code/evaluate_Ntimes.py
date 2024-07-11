import os
import numpy as np
import argparse
from medpy import metric
from tqdm import tqdm

from utils import read_list, read_nifti
from utils import config
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from scipy import stats

def cal_95CI(data):
    # Calculate sample statistics
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    sample_size = len(data)

    # Calculate t-value for 95% confidence interval
    t_value = stats.t.ppf(0.975, df=sample_size - 1)

    # Calculate margin of error
    margin_of_error = t_value * sample_std / np.sqrt(sample_size)

    # Calculate confidence interval
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    # Print the confidence interval
    print("95% Confidence Interval:")
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="fully")
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--cps', type=str, default=None)
    args = parser.parse_args()

    ids_list = read_list('test')
    results_all_folds = []

    txt_path = "./logs/"+args.exp+"/evaluation_res.txt"
    # print(txt_path)
    print("\n Evaluating...")
    fw = open(txt_path, 'w')
    for fold in range(1, args.folds+1):

        test_cls = [i for i in range(1, config.num_cls)]
        values = np.zeros((len(ids_list), len(test_cls), 2)) # dice and asd

        for idx, data_id in enumerate(tqdm(ids_list)):
            # if idx > 2:
            #     break
            print(data_id)
            pred = read_nifti(os.path.join("./logs",args.exp, "fold"+str(fold), "predictions_"+args.cps,f'{data_id}.nii.gz'))
            label = read_nifti(os.path.join(config.save_dir, 'processed', f'{data_id}_label.nii.gz')).astype(np.uint8)
            image = read_nifti(os.path.join(config.save_dir, 'processed', f'{data_id}_image.nii.gz')).astype(np.float32)

            dd, ww, hh = label.shape
            # label = torch.FloatTensor(label).unsqueeze(0).unsqueeze(0)
            # label = F.interpolate(label, size=(dd, 288, 288),mode='nearest')
            # label = label.squeeze().numpy()

            # padding_flag = label.shape[0] <= config.patch_size[0] or \
            #                label.shape[1] <= config.patch_size[1] or \
            #                label.shape[2] <= config.patch_size[2]
            # if padding_flag:
            #     pw = max((config.patch_size[0] - label.shape[0]) // 2 + 3, 0)
            #     ph = max((config.patch_size[1] - label.shape[1]) // 2 + 3, 0)
            #     pd = max((config.patch_size[2] - label.shape[2]) // 2 + 3, 0)
            #     # if padding_flag:
            #     label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            #     image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            save = sitk.GetImageFromArray(image)
            sitk.WriteImage(save, os.path.join("./logs",args.exp, "fold"+str(fold), "predictions_"+args.cps,f'{data_id}_image.nii.gz'))
            save = sitk.GetImageFromArray(label)
            sitk.WriteImage(save, os.path.join("./logs",args.exp, "fold"+str(fold), "predictions_"+args.cps,f'{data_id}_label.nii.gz'))


            # print(pred.shape)
            # print(label.shape)

            for i in test_cls:
                pred_i = (pred == i)
                label_i = (label == i)
                if pred_i.sum() > 0 and label_i.sum() > 0:
                    dice = metric.binary.dc(pred == i, label == i) * 100
                    hd95 = metric.binary.asd(pred == i, label == i)
                    values[idx][i-1] = np.array([dice, hd95])
                elif pred_i.sum() > 0 and label_i.sum() == 0:
                    dice, hd95 = 0, 128
                elif pred_i.sum() == 0 and label_i.sum() > 0:
                    dice, hd95 =  0, 128
                elif pred_i.sum() == 0 and label_i.sum() == 0:
                    dice, hd95 =  1, 0

                values[idx][i-1] = np.array([dice, hd95])

            print(values.shape)
        # values /= len(ids_list)
        values_mean_cases = np.mean(values, axis=0)
        results_all_folds.append(values)
        fw.write("Fold" + str(fold) + '\n')
        fw.write("------ Dice ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:,0],1)) + '\n')
        fw.write("------ ASD ------" + '\n')
        fw.write(str(np.round(values_mean_cases[:,1],1)) + '\n')
        fw.write('Average Dice:'+str(np.mean(values_mean_cases, axis=0)[0]) + '\n')
        fw.write('Average  ASD:'+str(np.mean(values_mean_cases, axis=0)[1]) + '\n')
        fw.write("=================================")
        print("Fold", fold)
        print("------ Dice ------")
        print(np.round(values_mean_cases[:,0],1))
        print("------ ASD ------")
        print(np.round(values_mean_cases[:,1],1))
        print(np.mean(values_mean_cases, axis=0)[0], np.mean(values_mean_cases, axis=0)[1])

    results_all_folds = np.array(results_all_folds)

    print(results_all_folds.shape)

    fw.write('\n\n\n')
    fw.write('All folds' + '\n')

    results_folds_mean = results_all_folds.mean(0)
    # print(f"\033[92m {results_folds_mean.mean(1)[:, 0]} \033[0m")


    patient_std = np.std(results_folds_mean.mean(1)[:, 0])
    patient_cls1_std = np.std(results_folds_mean[:, 0, 0])
    patient_cls2_std = np.std(results_folds_mean[:, 1, 0])
    print(f"\033[92m patient_cls1_std {patient_cls1_std} \033[0m")
    print(f"\033[92m patient_cls2_std {patient_cls2_std} \033[0m")
    print(f"\033[92m patient_std {patient_std} \033[0m")


    cal_95CI(results_folds_mean.mean(1)[:, 0])


    for i in range(results_folds_mean.shape[0]):
        fw.write("="*5 + " Case-" + str(ids_list[i]) + '\n')
        fw.write('\tDice:'+str(np.round(results_folds_mean[i][:,0],2).tolist()) + '\n')
        fw.write('\t ASD:'+str(np.round(results_folds_mean[i][:,1],2).tolist()) + '\n')
        fw.write('\t'+'Average Dice:'+str(np.mean(results_folds_mean[i], axis=0)[0]) + '\n')
        fw.write('\t'+'Average  ASD:'+str(np.mean(results_folds_mean[i], axis=0)[1]) + '\n')

    fw.write("=================================\n")
    fw.write('Final Dice of each class\n')
    fw.write(str([round(x,1) for x in results_folds_mean.mean(0)[:,0].tolist()]) + '\n')
    fw.write('Final ASD of each class\n')
    fw.write(str([round(x,1) for x in results_folds_mean.mean(0)[:,1].tolist()]) + '\n')
    print("=================================")
    print('Final Dice of each class')
    print(str([round(x,3) for x in results_folds_mean.mean(0)[:,0].tolist()]))
    print('Final ASD of each class')
    print(str([round(x,3) for x in results_folds_mean.mean(0)[:,1].tolist()]))
    std_dice = np.std(results_all_folds.mean(1).mean(1)[:,0])
    std_hd = np.std(results_all_folds.mean(1).mean(1)[:,1])

    fw.write('Final Avg Dice: '+str(round(results_folds_mean.mean(0).mean(0)[0], 3)) +'±' +  str(round(std_dice,2)) + '\n')
    fw.write('Final Avg  ASD: '+str(round(results_folds_mean.mean(0).mean(0)[1], 3)) +'±' +  str(round(std_hd,2)) + '\n')

    print('Final Avg Dice: '+str(round(results_folds_mean.mean(0).mean(0)[0], 3)) +'±' +  str(round(std_dice,2)))
    print('Final Avg  ASD: '+str(round(results_folds_mean.mean(0).mean(0)[1], 3)) +'±' +  str(round(std_hd,2)))



