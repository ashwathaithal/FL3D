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
canvas["x_max"] = 100
canvas["y_min"] = 0
canvas["y_max"] = 100
canvas["X_anchors"] = [10, 15, 20, 30, 35, 40, 50, 55, 60, 70, 75, 80]
if cfg.DETECT_OBJ == "Car":
    tag_det = "car_detection"
    tag_ori = "car_orientation"
    tag_det_grd = "car_detection_ground"
    tag_det_3d = "car_detection_3d"
elif cfg.DETECT_OBJ == "Pedestrian":
    tag_det = "pedestrian_detection"
    tag_ori = "pedestrian_orientation"
    tag_det_grd = "pedestrian_detection_ground"
    tag_det_3d = "pedestrian_detection_3d"
elif cfg.DETECT_OBJ == "Cyclist":
    tag_det = "cyclist_detection"
    tag_ori = "cyclist_orientation"
    tag_det_grd = "cyclist_detection_ground"
    tag_det_3d = "cyclist_detection_3d"
else:
    raise NotImplementedError

ytick_det = cfg.DETECT_OBJ[:3]+"_"+"det"
ytick_ori = cfg.DETECT_OBJ[:3]+"_"+"ori"
ytick_det_grd = cfg.DETECT_OBJ[:3]+"_"+"det_grd"
ytick_det_3d = cfg.DETECT_OBJ[:3]+"_"+"det_3d"

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
        if line.startswith(tag_det + " AP:"):
            cmd_dict[tag_det] = [float(res) for res in line.split(' ')[-3:]] 
        elif line.startswith(tag_ori + " AP:"):
            cmd_dict[tag_ori] = [float(res) for res in line.split(' ')[-3:]] 
        elif line.startswith(tag_det_grd + " AP:"):
            cmd_dict[tag_det_grd] = [float(res) for res in line.split(' ')[-3:]]
        elif line.startswith(tag_det_3d + " AP:"):
            cmd_dict[tag_det_3d] = [float(res) for res in line.split(' ')[-3:]]
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
    if tag_det in res.keys():
        results.append(res)

markers = []
mks = itertools.cycle(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'))
for res in results:
    Y = res[tag_det] + res[tag_ori] + res[tag_det_grd] + res[tag_det_3d]
    C = (random.random(), random.random(), random.random())
    marker = plt.scatter(canvas["X_anchors"], Y, c=C, s=25, alpha=1, label=res["tag"], marker = mks.__next__())
    markers.append(marker)
plt.xlim(canvas["x_min"], canvas["x_max"])
plt.ylim(canvas["y_min"])
plt.legend(handles=markers, ncol=1)
plt.xticks([canvas["X_anchors"][0], canvas["X_anchors"][3], canvas["X_anchors"][6], canvas["X_anchors"][9]], [ytick_det, ytick_ori, ytick_det_grd, ytick_det_3d] )
plt.legend(bbox_to_anchor=(0.85,1), loc="upper left")
plt.savefig(os.path.join(predall_dir, "kitti_eval.png"))