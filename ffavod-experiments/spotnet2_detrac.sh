#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

python test.py ctdetSpotNet2 --test --exp_id reproduceTopDetrac --dataset uadetrac1on10_b --arch hourglassSpotnet2 --keep_res --load_model /store/datasets/UA-DetracResults/exp/ctdet/SpotNet2_ADD_DB_PYFLOW_LR2e6/model_best.pth

