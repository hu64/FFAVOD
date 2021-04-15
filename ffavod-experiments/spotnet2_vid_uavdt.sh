#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

# python main.py ctdetSpotNetVid --val_intervals 1 --exp_id fromCOCO --nbr_frames 5 --elliptical_gt --seg_weight 0.1 --dataset uav --arch hourglassSpotNetVid --batch_size 1 --lr 6.25e-5 --load_model ../models/ctdet_coco_hg.pth
# python test.py --test ctdetSpotNetVid --exp_id fromCOCO --nbr_frames 5 --dataset uav --arch hourglassSpotNetVid --keep_res --load_model ../exp/uav/ctdetSpotNetVid/fromCOCO/model_best.pth
