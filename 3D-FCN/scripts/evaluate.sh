#!/bin/bash
TAG=$1
python3 test.py
./kitti_eval/evaluate_object_3d_offline /usr/app/TuneDataKitti/validation/label_2/ ./predicts/$TAG/ > ./predicts/$TAG/cmd.log