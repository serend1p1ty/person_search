# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import torch

from utils.config import cfg


def generate_anchors():
    """
    Generate anchors by enumerating aspect ratios and
    scales wrt a reference (0, 0, 15, 15) window.
    """
    ratios = torch.Tensor(cfg.ANCHOR_RATIOS)
    scales = torch.Tensor(cfg.ANCHOR_SCALES)
    base_anchor = torch.Tensor([0, 0, 15, 15])
    ratio_anchors = ratio_enum(base_anchor, ratios)
    anchors = torch.cat([scale_enum(anchor, scales) for anchor in ratio_anchors], dim=0)
    return anchors


def whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor.
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around
    a center (x_ctr, y_ctr), output a set of anchors.
    """
    ws.unsqueeze_(1)
    hs.unsqueeze_(1)
    anchors = torch.cat(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        ),
        dim=1,
    )
    return anchors


def ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for a set of aspect ratios wrt an anchor.
    """
    w, h, x_ctr, y_ctr = whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = torch.round(torch.sqrt(size_ratios))
    hs = torch.round(ws * ratios)
    anchors = mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for a set of scales wrt an anchor.
    """
    w, h, x_ctr, y_ctr = whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
