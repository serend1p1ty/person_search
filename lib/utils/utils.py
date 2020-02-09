import logging
import os
import os.path as osp
import pickle as pk

import coloredlogs
import torch

from utils.config import cfg


def smooth_l1_loss(deltas, gt_deltas, inside_weights, outside_weights, sigma=1):
    """
    Calculate smooth L1 loss introduced by Fast-RCNN.

    f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
           |x| - 0.5 / sigma / sigma    otherwise

    Args:
        deltas (Tensor): Predicted regression deltas.
        gt_deltas (Tensor): Ground-truth regression deltas.
        inside_weights (Tensor): Calculate loss only for foreground proposals.
        outside_weights (Tensor): Weights of the smooth L1 loss relative to classification loss.
        sigma (float): Super parameter of smooth L1 loss.

    Returns:
        loss (Tensor)
    """
    sigma_2 = sigma ** 2
    x = inside_weights * torch.abs(deltas - gt_deltas)
    loss = torch.where(x < 1 / sigma_2, x ** 2 * sigma_2 * 0.5, x - 0.5 / sigma_2)
    loss *= outside_weights
    loss = loss.sum() / loss.size(0)
    return loss


def torch_rand_choice(arr, size):
    """
    Generates a random sample from a given array, like numpy.random.choice.
    """
    idxs = torch.randperm(arr.size(0))[:size]
    return arr[idxs]


def pickle(data, file_path):
    with open(file_path, "wb") as f:
        pk.dump(data, f, pk.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, "rb") as f:
        data = pk.load(f)
    return data


def init_logger(file_name, level="INFO"):
    """
    Initialize the colored logger and save the log to a file.
    """
    log_dir = osp.join(cfg.DATA_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = osp.join(log_dir, file_name)
    fmt_str = "[%(asctime)s] [%(filename)-12s] [%(levelname)s] : %(message)s"
    logging.basicConfig(filename=log_file, format=fmt_str)
    coloredlogs.install(level=level, fmt=fmt_str)
