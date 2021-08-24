#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

python test.py --test ctdetSpotNet2 --exp_id fromPaper --dataset uav --arch hourglassSpotnet2 --keep_res --load_model /store/dev/SpotNet2/exp/uav/ctdetSpotNetVid/fromPaper/uavdt_model_best.pth  # /store/dev/SpotNet2/exp/uav/ctdetSpotNet2/fromPaper/model_best.pth
# python test.py --test ctdetSpotNetVid --exp_id fromPaper --nbr_frames 5 --dataset uav --arch hourglassSpotNetVid --keep_res --load_model /store/dev/SpotNet2/exp/uav/ctdetSpotNetVid/fromPaper/uavdt_model_best.pth  # /store/dev/SpotNet2/exp/uav/ctdetSpotNet2/fromPaper/model_best.pth


