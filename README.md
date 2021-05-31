# Person Search

## :sparkles: News: We release the [source code](https://github.com/serend1p1ty/SeqNet.git) of the current state-of-the-art model [SeqNet(AAAI 2021)](https://arxiv.org/abs/2103.10148), which achieves :trophy: `94.8%` mAP on CUHK-SYSU.

## Introduction

A pytorch implementation for CVPR 2017 "Joint Detection and Identification Feature Learning for Person Search".

The code is based on the [offcial caffe version](https://github.com/ShuangLI59/person_search.git).

You can find a better one achieving about `85%` mAP in `mmdetection` branch!

**Note**: The implementaion of Faster R-CNN in `mmdetection` branch is better than that described in original paper.

## Highlights

- **Simpler code**: After reduction and refactoring, the current version is simpler and easier to understand.
- **Pure Pytorch code**: Numpy is not used, except for data loading.

## Installation

Run `pip install -r requirements.txt` in the root directory of the project

`torchvision` must be greater than 0.3.0, as we need `torchvision.ops.nms`

## Quick Start

Let's say `$ROOT` is the root directory.

1. Download CUHK-SYSU ([google drive](https://drive.google.com/open?id=1z3LsFrJTUeEX3-XjSEJMOBrslxD2T5af) or [baiduyun](https://pan.baidu.com/s/1jHLfeZk)) dataset, unzip to `$ROOT/data/dataset/`
2. Download our trained model ([google drive](https://drive.google.com/open?id=1ta6YfttPLsMSiip3sn9TqzOeSTG4ASdd) or [baiduyun](https://pan.baidu.com/s/1myLvpWHWJcAne3xDVuvQGg)) (extraction code: `uuti`) to `$ROOT/data/trained_model/`

After the above two steps, the directory structure should look like this:

```
$ROOT/data
├── dataset
│   ├── annotation
│   ├── Image
│   └── README.txt
└── trained_model
    └── checkpoint_step_50000.pth
```

BTW, `$ROOT/data` saves all experimental data, include: dataset, pretrained model, trained model, and so on.

3. Run `python tools/demo.py --gpu 0 --checkpoint data/trained_model/checkpoint_step_50000.pth`.
   And then you can checkout the result in `imgs` directory.

![demo.jpg](./imgs/demo.jpg)

## Train

1. Prepare dataset as described in **Quick Start** section.
2. Download pretrained model ([google drive](https://drive.google.com/open?id=1vFDwjG12WC43Blo6ea_TZASDQr0lvxiM) or [baiduyun](https://pan.baidu.com/s/1dC8dEuB_8pV8m6Msrj8dXw)) (extraction code `ucnw`) to `$ROOT/data/pretrained_model/`
3. `python tools/train_net.py --gpu 0`
4. Trained model will be saved to `$ROOT/data/trained_model/`

You can check the usage of `train_net.py` by running `python tools/train_net.py -h`

## Test

`python tools/test_net.py --gpu 0 --checkpoint data/trained_model/checkpoint_step_50000.pth`

The result should be around:

```
Search ranking:
   mAP = 76.78%
   Top- 1 = 77.48%
   Top- 5 = 88.48%
   Top-10 = 91.52%
```
