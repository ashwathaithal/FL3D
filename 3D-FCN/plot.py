import os
import random
import matplotlib
import glob
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import itertools
from config import cfg

canvas = dict()
canvas["x_min"] = 0
canvas["x_max"] = 200
canvas["y_min"] = 0
canvas["y_max"] = 100
canvas["X_anchors"] = [10, 15, 20, 30, 35, 40, 50, 55, 60, 70, 75, 80]

def get_ckpt_list(ckpt_dir):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, "checkpoint-*"))
    ckpt_list = [ckpt.split('/')[-1] for ckpt in ckpt_list]
    ckpt_list = [ckpt.split('.')[0] for ckpt in ckpt_list]
    return ckpt_list

def load_kitti_eval_log(log_path):
    if not os.path.exists(log_path):
        return dict()
    f = open(log_path, "r")
    lines = f.readlines()
    cmd_dict = dict()
    for line in lines:
        line = line.rstrip()
        if line.startswith("car_detection AP:"):
            cmd_dict["car_detection"] = [float(res) for res in line.split(' ')[-3:]] 
        elif line.startswith("car_orientation AP:"):
            cmd_dict["car_orientation"] = [float(res) for res in line.split(' ')[-3:]] 
        elif line.startswith("car_detection_ground AP:"):
            cmd_dict["car_detection_ground"] = [float(res) for res in line.split(' ')[-3:]]
        elif line.startswith("car_detection_3d AP:"):
            cmd_dict["car_detection_3d"] = [float(res) for res in line.split(' ')[-3:]]
        else:
            continue
    return cmd_dict

tag = cfg.TAG
predall_dir = os.path.join("./predicts-all", tag)
ckpt_list = get_ckpt_list(predall_dir)
ckpt_list.sort()

results = []
for ckpt in ckpt_list:
    print(ckpt)
    pred_dir = os.path.join(predall_dir, ckpt)
    cmd_path = os.path.join(pred_dir, "cmd.log")
    res = load_kitti_eval_log(cmd_path)
    res["tag"] = ckpt.split("-")[-1]
    if "car_detection" in res.keys():
        results.append(res)

markers = []
mks = itertools.cycle(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'))
for res in results:
    Y = res["car_detection"] + res["car_orientation"] + res["car_detection_ground"] + res["car_detection_3d"]
    C = (random.random(), random.random(), random.random())
    marker = plt.scatter(canvas["X_anchors"], Y, c=C, s=25, alpha=1, label=res["tag"], marker = mks.__next__())
    markers.append(marker)
plt.xlim(canvas["x_min"], canvas["x_max"])
plt.ylim(canvas["y_min"], )
plt.legend(handles=markers, ncol=2)
plt.xticks([canvas["X_anchors"][0], canvas["X_anchors"][3], canvas["X_anchors"][6], canvas["X_anchors"][9]], ['car_det', 'car_ori', 'car_det_grd', 'car_det_3d'] )
plt.savefig(os.path.join(predall_dir, "kitti_eval.png"))