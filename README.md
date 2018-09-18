# README
This repository is for "Focal Loss in 3D Object Detection".
If you use this code, we would appreciate if you cite our paper:
```
@article{yun2018fl3d,
  title={Focal Loss in 3D Object Detection},
  author={Peng Yun, Lei Tai, Yuan Wang, and Ming Liu},
  journal={arXiv preprint arXiv:1809.06065},
  year={2018}
}
```
## Organization
It contains two parts: 3D-FCN and VoxelNet as claimed in our paper.
For detailed information of these two parts, please refer the sub-README in each dir.
## Data & Weights
The splited data & weights are available in [Data&Weights](https://hkustconnect-my.sharepoint.com/:f:/g/personal/pyun_connect_ust_hk/EvB4NEzNw7xGqSGVGoJyWIgBCiTNUhZYOC30stN0xpTnCg?e=1KaHDp)
## Docker
The docker images for 3D-FCN and VoxelNet are also available.
```
docker pull pyun/voxelnet:baseline
docker pull pyun/3d-fcn:tf1.7
# Then launch a container with nvidia-docker 1.0
```
