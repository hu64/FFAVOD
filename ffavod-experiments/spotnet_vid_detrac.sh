#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

# reproduce old detrac result
# python test.py ctdetSpotNet2 --exp_id fromPaper --dataset uadetrac1on10_b --arch hourglassSpotnet2 --keep_res --load_model /store/datasets/UA-DetracResults/SpotNet/ua-detrac_model_best.pth
# with vid
python test.py ctdetSpotNetVid --exp_id fromPaper --nbr_frames 5 --dataset uadetrac1on10_b --arch hourglassSpotNetVid --keep_res --load_model /store/datasets/UA-DetracResults/SpotNet/ua-detrac_model_best.pth
