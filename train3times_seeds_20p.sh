#!/bin/bash


while getopts 'e:c:t:l:w:' OPT; do
    case $OPT in
        e) exp=$OPTARG;;
        c) cuda=$OPTARG;;
		    t) task=$OPTARG;;
		    l) lr=$OPTARG;;
		    w) cps_w=$OPTARG;;

    esac
done
echo $exp
echo $cuda

epoch=300
echo $epoch

labeled_data="labeled"
unlabeled_data="unlabeled"
folder="Exp_AB/"
cps="AB"


#python code/train_${exp}.py --exp ${folder}${exp}${task}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} -r
#python code/test.py --exp ${folder}${exp}${task}/fold1 -g ${cuda} --cps ${cps} --split test_new
python code/evaluate_Ntimes.py --exp ${folder}${exp}${task} --folds 1 --cps ${cps}
#python code/test.py --exp ${folder}${exp}${task}/fold1 -g ${cuda} --cps ${cps} --split test_new

#python code/SAM_refine.py --exp ${folder}${exp}${task} --folds 1 --cps ${cps}
#python code/train_${exp}.py --exp ${folder}${exp}${task}/fold2 --seed 1 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} -r
#python code/test.py --exp ${folder}${exp}${task}/fold2 -g ${cuda} --cps ${cps}
#python code/evaluate_Ntimes.py --exp ${folder}${exp}${task} --folds 2 --cps ${cps}
#python code/train_${exp}.py --exp ${folder}${exp}${task}/fold3 --seed 666 -g ${cuda} --base_lr ${lr} -w ${cps_w} -ep ${epoch} -sl ${labeled_data} -su ${unlabeled_data} -r
#python code/test.py --exp ${folder}${exp}${task}/fold3 -g ${cuda} --cps ${cps}

#python code/evaluate_Ntimes.py --exp ${folder}${exp}${task} --cps ${cps}
