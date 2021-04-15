#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

# python run_on_csv_method.py --task 'ctdetSpotNetVid' --exp_id '3F' --n_frames 3 --model_name 'model_best.pth'
# python csv_to_detrac.py --name 'SNVID_3F' --base_dir '../exp/uadetrac1on10_b/ctdetSpotNetVid/3F'

#python run_on_csv_method.py --task 'ctdetSpotNetVid' --exp_id '7F' --n_frames 7 --model_name 'model_best.pth'
#python csv_to_detrac.py --name 'SNVID_7F' --base_dir '../exp/uadetrac1on10_b/ctdetSpotNetVid/7F'
#
#python run_on_csv_method.py --task 'ctdetSpotNetVid' --exp_id '9F' --n_frames 9 --model_name 'model_best.pth'
#python csv_to_detrac.py --name 'SNVID_9F' --base_dir '../exp/uadetrac1on10_b/ctdetSpotNetVid/9F'

# python run_on_csv_method.py --task 'ctdetSpotNetVid' --exp_id '11F' --n_frames 11 --model_name 'model_best.pth'
# python csv_to_detrac.py --name 'SNVID_11F' --base_dir '../exp/uadetrac1on10_b/ctdetSpotNetVid/11F'

#python run_on_csv_method.py --task 'ctdetSpotNetVid' --exp_id '13F' --n_frames 13 --model_name 'model_best.pth'
#python csv_to_detrac.py --name 'SNVID_13F' --base_dir '../exp/uadetrac1on10_b/ctdetSpotNetVid/13F'

# python run_on_csv_method.py --task 'ctdetSpotNetVid' --exp_id '5Conc' --n_frames 5 --model_name 'model_best.pth'
# python csv_to_detrac.py --name 'SNVID_5Conc' --base_dir '../exp/uadetrac1on10_b/ctdetSpotNetVid/5Conc'

#python run_on_csv_method.py --task 'ctdetSpotNetVid' --exp_id '5Prec' --n_frames 5 --model_name 'model_best.pth'
#python csv_to_detrac.py --name 'SNVID_5Prec' --base_dir '../exp/uadetrac1on10_b/ctdetSpotNetVid/5Prec'

# python run_on_csv_method.py --task 'ctdetSpotNet2' --exp_id 'spotnet2_fromDetrac' --model_name 'model_last.pth'
# python run_on_csv_method.py --task 'ctdetSpotNet2' --exp_id 'spotnet2_fromUAV' --model_name 'model_best.pth'
# python run_on_csv_method.py --task 'ctdetSpotNet2' --exp_id 'spotnet2_fromUAVB' --model_name 'model_best.pth'
# python run_on_csv_method.py --task 'ctdetSpotNet2' --exp_id 'fromSNVID' --model_name 'BBfromSN.pth'

# python run_on_csv_method.py --task 'ctdet' --exp_id 'reproduceOlder' --model_name 'model_best.pth'
# python run_on_csv_method.py --task 'ctdetSpotNet2' --exp_id 'fromCOCO' --model_name 'model_best.pth'
# python run_on_csv_method.py --task 'ctdetSpotNet2' --exp_id 'fromCOCOB' --model_name 'model_best.pth'
# python run_on_csv_method.py --task 'ctdetSpotNetVid' --exp_id 'fromCOCOB' --n_frames 5 --model_name 'fromSN2.pth'
# python run_on_csv_method.py --task 'ctdetSpotNetVid' --exp_id 'fromCOCO' --n_frames 5 --model_name 'model_best.pth'