import sys
import os
CENTERNET_PATH = '/store/dev/SpotNet2/src/lib/' if os.path.exists('/store/dev/SpotNet2/src/lib/') else '/home/travail/huper/dev/SpotNet2/src/lib/'
# CENTERNET_PATH = '/store/dev/CenterNet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts
import os
import cv2
import numpy as np
from PIL import Image
import getopt
from tqdm import tqdm
# class_name = ['__background__', 'bus', 'car', 'others', 'van']
class_name = ['__background__', 'object']


def main(argv):
    try:
        opts0, args = getopt.getopt(argv, "ht:e:n:m", ["task=", "exp_id=", "n_frames=", "model_name="])
    except getopt.GetoptError:
        # print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts0:
        if opt == '-h':
            # print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-t", "--task"):
            task = arg
            print('task: ', task)
        elif opt in ("-e", "--exp_id"):
            exp_id = arg
            print('exp_id: ', exp_id)
        elif opt in ("-n", "--n_frames"):
            n_frames = arg
            print('n_frames: ', n_frames)
        elif opt in ("-m", "--model_name"):
            model_name = arg
            print('model_name: ', model_name)

    # base_dir = os.path.join('/store/dev/SpotNet2/exp/uadetrac1on10_b/', task) if os.path.exists(os.path.join('/store/dev/SpotNet2/exp/uadetrac1on10_b/', task)) \
    #     else os.path.join('/home/travail/huper/dev/SpotNet2/exp/uadetrac1on10_b/', task)
    base_dir = os.path.join('/usagers2/huper/dev/SpotNet2/exp/uav/', task)
    model_path = os.path.join(base_dir, exp_id, model_name)
    # model_path = '/store/datasets/UA-DetracResults/exp/ctdet/UAV-CTNET-STD/model_best.pth'

    opt = opts().init((task + ' --arch hourglassSpotNetVid --nbr_frames ' + n_frames + ' --dataset uav --keep_res --load_model ' + model_path).split(' '))
    # opt = opts().init((task + ' --arch hourglassSpotnet2 --dataset uav --keep_res --load_model ' + model_path).split(' '))
    # opt = opts().init((task + ' --arch hourglass --dataset uav --keep_res --load_model ' + model_path).split(' '))

    print(opt)
    detector = detector_factory[opt.task](opt)
    DATASET_DIR = '/store/datasets/UA-Detrac/' if os.path.exists('/store/datasets/UA-Detrac/') \
        else '/home/travail/huper/datasets/UA-Detrac/'

    SPLIT = 'uav-test'

    if SPLIT == 'test':
        source_lines = open(os.path.join(DATASET_DIR, 'test-tf-all.csv'), 'r').readlines()
        target_file = open(os.path.join(base_dir, exp_id, 'ua-test.csv'), 'a')
    elif SPLIT == 'train1on10':
        source_lines = open(os.path.join(DATASET_DIR, 'train-tf.csv'), 'r').readlines() # + open('/store/datasets/UA-Detrac/val-tf-all.csv', 'r').readlines()
        target_file = open(os.path.join(base_dir, exp_id, 'ua-train1on10.csv'), 'w')
    elif SPLIT == 'trainval':
        source_lines = open(os.path.join(DATASET_DIR, 'train-tf-all.csv'), 'r').readlines() + open('/store/datasets/UA-Detrac/val-tf-all.csv', 'r').readlines()
        target_file = open(os.path.join(base_dir, exp_id, 'ua-trainval.csv'), 'w')
    elif SPLIT == 'uav-test':
        source_lines = open('/store/datasets/UAV/val.csv', 'r').readlines()
        target_file = open(os.path.join(base_dir, exp_id, 'uav-test.csv'), 'w')
    elif SPLIT == 'uav-val':
        source_lines = open('/store/datasets/UAV/val-sub.csv', 'r').readlines()
        target_file = open(os.path.join(base_dir, exp_id, 'uav-val2.csv'), 'w')
    elif SPLIT == 'changedetection':
        source_lines = open('/store/datasets/changedetection/changedetection.csv', 'r').readlines()
        target_file = open(os.path.join(base_dir, exp_id, 'changedetection.csv'), 'w')
    elif SPLIT == 'ped1':
        source_lines = open('/store/datasets/ped1/csv.csv', 'r').readlines()
        target_file = open(os.path.join(base_dir, exp_id, 'results.csv'), 'w')

    images = [item.split(',')[0] for item in source_lines]
    images = sorted(list(set(images)))

    # remove_lines = open('/usagers2/huper/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/11F/ua-test.csv', 'r').readlines()
    # remove_images = [item.split(',')[0] for item in remove_lines]
    # remove_images = set(remove_images)
    #
    # for remove_image in remove_images:
    #     images.remove(remove_image)

    for img in tqdm(images):
        img_path = img.strip()
        if opt.task == 'ctdetVid' or opt.task == 'ctdetSpotNetVid':
            n_frames = int(n_frames)
            middle = int(n_frames / 2)
            index = os.path.basename(img_path).replace('.jpg', '').replace('img', '').replace('.JPEG', '')
            rest = img_path.replace(index + '.jpg', '').replace(os.path.dirname(img_path), '')
            length = len(index)
            modulo = '1'
            for i in range(length):
                modulo += '0'
            img_paths = []
            for i in range(n_frames):
                new_img_path = os.path.dirname(img_path) \
                               + rest \
                               + str((int(index) - (i - middle)) % int(modulo)).zfill(length) + '.jpg'
                if not os.path.exists(new_img_path):
                    new_img_path = img_path
                img_paths.append(new_img_path)
            imgs = []
            for path in img_paths:
                loaded_img = cv2.imread(path)
                imgs.append(loaded_img)
            im = np.concatenate(imgs, -1)
        else:
            im = np.array(Image.open(img_path))
        runRet = detector.run(im)
        ret = runRet['results']
        for label in [1]:  # , 2, 3, 4]:
            for det in ret[label]:
                box = [int(item) for item in det[:4]]
                det = [img.strip()] + box + [class_name[label]] + [det[4]]
                print(str(det)[1:-1].translate(str.maketrans('', '', '\' ')), file=target_file)


if __name__ == "__main__":
   main(sys.argv[1:])

