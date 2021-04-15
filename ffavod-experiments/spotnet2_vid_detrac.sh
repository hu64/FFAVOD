#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src


# python main.py ctdetSpotNetVid --val_intervals 1 --exp_id fromDetrac --dataset uadetrac1on10_b --arch hourglassSpotNetVid --batch_size 1 --master_batch 4 --lr 2e-5 --load_model /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromSpotNet2/FusionFromDetrac.pth
# python main.py ctdetSpotNetVid --val_intervals 1 --exp_id fromDetrac --dataset uadetrac1on10_b --arch hourglassSpotNetVid --batch_size 1 --master_batch 4 --lr 2e-5 --resume
python test.py ctdetSpotNetVid --test --exp_id fromDetrac --nbr_frames 5 --dataset uadetrac1on10_b --arch hourglassSpotNetVid --keep_res --load_model /usagers2/huper/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_best.pth # /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/AblationTestfromDetrac3/model_best.pth

# python main.py ctdetSpotNetVid --val_intervals 1 --nbr_frames 3 --num_epochs 1 --exp_id 3F --dataset uadetrac1on10_b --arch hourglassSpotNetVid --batch_size 1 --master_batch 4 --lr 2e-7 --load_model /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_1.pth  # /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_best.pth
# python test.py ctdetSpotNetVid --test --exp_id 3F --nbr_frames 3 --dataset uadetrac1on10_b --arch hourglassSpotNetVid --keep_res --load_model  /usagers2/huper/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/3F/model_best.pth # /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/AblationTestfromDetrac3/model_best.pth
#python main.py ctdetSpotNetVid --val_intervals 1 --nbr_frames 7 --num_epochs 1 --exp_id 7F --dataset uadetrac1on10_b --arch hourglassSpotNetVid --batch_size 1 --master_batch 4 --lr 2e-7 --load_model /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_1.pth  # /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_best.pth
#python main.py ctdetSpotNetVid --val_intervals 1 --nbr_frames 9 --num_epochs 1 --exp_id 9F --dataset uadetrac1on10_b --arch hourglassSpotNetVid --batch_size 1 --master_batch 4 --lr 2e-7 --load_model /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_1.pth  # /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_best.pth
#python main.py ctdetSpotNetVid --val_intervals 1 --nbr_frames 11 --num_epochs 1 --exp_id 11F --dataset uadetrac1on10_b --arch hourglassSpotNetVid --batch_size 1 --master_batch 4 --lr 2e-7 --load_model /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_1.pth  # /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_best.pth
#python main.py ctdetSpotNetVid --val_intervals 1 --nbr_frames 13 --num_epochs 1 --exp_id 13F --dataset uadetrac1on10_b --arch hourglassSpotNetVid --batch_size 1 --master_batch 4 --lr 2e-7 --load_model /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_1.pth  # /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_best.pth

# python main.py ctdetSpotNetVid --val_intervals 1 --nbr_frames 5 --num_epochs 1 --exp_id 5Conc --dataset uadetrac1on10_b --arch hourglassSpotNetVid --batch_size 1 --master_batch 4 --lr 2e-7 --load_model /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_1.pth  # /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_best.pth
# python test.py ctdetSpotNetVid --test --exp_id 5Conc --nbr_frames 5 --dataset uadetrac1on10_b --arch hourglassSpotNetVid --keep_res --load_model  /usagers2/huper/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/5Conc/model_best.pth # /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/AblationTestfromDetrac3/model_best.pth

# python main.py ctdetSpotNetVid --val_intervals 1 --nbr_frames 5 --num_epochs 1 --exp_id 5Prec --dataset uadetrac1on10_b --arch hourglassSpotNetVid --batch_size 1 --master_batch 4 --lr 2e-6 --load_model /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_1.pth
# python test.py ctdetSpotNetVid --test --exp_id 5Prec --nbr_frames 5 --dataset uadetrac1on10_b --arch hourglassSpotNetVid --keep_res --load_model  /store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/5Prec/model_best.pth
