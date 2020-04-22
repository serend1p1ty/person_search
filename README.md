# Person Search

## Introduction

A pytorch implementation for CVPR 2017 "Joint Detection and Identification Feature Learning for Person Search".

The code is based on the [offcial caffe version](https://github.com/ShuangLI59/person_search.git).

## Highlights

-   **Simpler code**: After reduction and reconstruction of the original code, the current version is simpler and easier to understand.
-   **Pure Pytorch code**: Numpy is not used, except for data reading.
-   **Good code style**: Linter (flake8) and formatter (black) are used to ensure code quality.

## Installation

Run `pip install -r requirements.txt` in the root directory of the project

`torchvision` must be greater than 0.3.0, as we need `torchvision.ops.nms`

## Quick Start

Let's say `$ROOT` is the root directory.

1. Download [CUHK-SYSU](https://pan.baidu.com/s/1jHLfeZk) dataset, unzip to `$ROOT/data/dataset/`
2. Download our [trained model](https://pan.baidu.com/s/1myLvpWHWJcAne3xDVuvQGg) (extraction code: uuti) to `$ROOT/data/trained_model/`

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
   And then you can checkout the result in `demo` directory.

![demo.jpg](./imgs/demo.jpg)

## Train

1. Prepare dataset as we mentioned in **Quick Start** section.
2. Download [pretrained model](https://pan.baidu.com/s/1pYkGhnpl46DCuKyIbNNXqQ) to `$ROOT/data/pretrained_model/`
3. `python tools/train_net.py --gpu 0`
4. Trained model will be saved to `$ROOT/data/trained_model/`

You can check the usage of `train_net.py` by running `python tools/train_net.py -h`

## Test

`python tools/test_net.py --gpu 0 --checkpoint data/trained_model/checkpoint_step_50000.pth`

## Future plans

-   Support for PRW dataset.
-   Re-implementation based on detectron2
