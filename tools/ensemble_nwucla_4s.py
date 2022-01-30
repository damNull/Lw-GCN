import argparse
import pickle

import numpy as np
from tqdm import tqdm
import os, sys

sys.path.extend([os.path.join(os.path.dirname(__file__), '../')])
from feeders.NWUCLA_feeder import Feeder

ds = Feeder('', label_path='val')

label = ds.data_dict
label = np.array([[i['file_name'] for i in label], [i['label'] - 1 for i in label]])

# test parameter logic
if len(sys.argv) == 5:
    print('use custom ensemble parameter')
    alpha = [float(i) for i in sys.argv[1:]]
else:
    alpha = [0.275, 0.325, 0.275, 0.125] 

root_dir = './evaluations/NW-UCLA'

r1 = open('%s/joint/results.pkl' % root_dir, 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('%s/bone/results.pkl' % root_dir, 'rb')
r2 = list(pickle.load(r2).items())
r3 = open('%s/joint_motion/results.pkl' % root_dir, 'rb')
r3 = list(pickle.load(r3).items())
r4 = open('%s/bone_motion/results.pkl' % root_dir, 'rb')
r4 = list(pickle.load(r4).items())

rs = [r1, r2, r3, r4]

right_num = total_num = right_num_5 = 0
for i in range(len(label[0])):
    _, l = label[:, i]
    r = [rs[j][i][1] for j in range(len(rs))]
    r = sum([r[j] * alpha[j] for j in range(len(rs))])
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print('top1: ', acc)
print('top5: ', acc5)
