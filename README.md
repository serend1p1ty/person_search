# Person Search

A pytorch implementation for CVPR 2017 "Joint Detection and Identification Feature Learning for Person Search" based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Requirements

mmdetection: 55a4023c (no guarantee for later versions)

## Training

Let's say `$ROOT` is the root directory.

1. Download CUHK-SYSU ([google drive](https://drive.google.com/open?id=1z3LsFrJTUeEX3-XjSEJMOBrslxD2T5af) or [baiduyun](https://pan.baidu.com/s/1jHLfeZk)) dataset, unzip to `$ROOT/data/cuhk_sysu`. Then the directory structure should look like this:

```
$ROOT/data
└── cuhk_sysu
    ├── annotation
    ├── cache
    ├── Image
    └── README.txt
```

2. `python train.py configs/cuhk_sysu/faster_rcnn_r50_caffe_c4_1x_cuhk.py --gpus 1 --no-validate`

## Test

`python test.py configs/cuhk_sysu/faster_rcnn_r50_caffe_c4_1x_cuhk.py ./work_dir/epoch_4.pth`

The result should be around:

```
All detection:
  Recall = 84.10%
  AP = 79.05%
Labeled only detection:
  Recall = 98.53%
Search ranking:
  mAP = 85.21%
  Top- 1 = 85.52%
  Top- 5 = 94.86%
  Top-10 = 96.86%
```
