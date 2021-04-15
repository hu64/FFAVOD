#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

# python main.py ctdetSpotNet2 --val_intervals 1 --exp_id fromCOCOB --elliptical_gt --seg_weight 0.1 --dataset uav --arch hourglassSpotnet2 --batch_size 1 --lr 6.25e-5 --load_model ../models/ctdet_coco_hg.pth
# python test.py --test ctdetSpotNet2 --exp_id fromCOCOB --dataset uav --arch hourglassSpotnet2 --keep_res --load_model ../exp/uav/ctdetSpotNet2/fromCOCOB/model_best.pth

