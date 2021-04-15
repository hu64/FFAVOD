import csv
import os
import numpy as np
import getopt
import sys
from tqdm import tqdm


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hn:b", ["name=", "base_dir="])
    except getopt.GetoptError:
        # print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            # print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-n", "--name"):
            algo_name = arg
            print('name: ', algo_name)
        elif opt in ("-b", "--base_dir"):
            base_dir = arg
            print('base_dir: ', base_dir)

    lines = sorted(open(os.path.join(base_dir, 'ua-test.csv'), 'r').readlines())
    res_dir = os.path.join(base_dir, algo_name)

    if not os.path.exists(res_dir):
            os.mkdir(res_dir)
    seq = {}

    for line in tqdm(lines):
        items = line.split(',')
        seq_name = os.path.dirname(items[0]).split('/')[-1]

        index = int(os.path.basename(items[0]).replace('.jpg', '').replace('img', ''))

        x0, y0, x1, y1 = [int(item) for item in items[1:5]]

        x0 = np.clip(x0, 0, 960)
        x1 = np.clip(x1, 0, 960)
        y0 = np.clip(y0, 0, 540)
        y1 = np.clip(y1, 0, 540)

        w, h = x1-x0+1, y1-y0+1

        score = float(items[6].strip())
        if score >= 0.0:
            if seq_name in seq:
                seq[seq_name].append([index, x0, y0, w, h, score])
            else:
                seq[seq_name] = [[index, x0, y0, w, h, score]]

    for sequence in seq:
        filename = os.path.join(res_dir, sequence + '_Det_' + algo_name + '.txt')
        writer = csv.writer(open(filename, 'w'), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        count = 1
        index = 0
        first = True
        for det in seq[sequence]:
            if first:
                index = det[0]
                first = False
            else:
                if index == det[0]:
                    count += 1
                else:
                    index = det[0]
                    count = 1

            writer.writerow([index, count, det[1], det[2], det[3], det[4], det[5]])


if __name__ == "__main__":
   main(sys.argv[1:])





