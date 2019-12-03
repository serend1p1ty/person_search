"""
Author: Ross Girshick and Sean Bell
Description: Assign labels and regression targets to anchors.
"""

import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn

from rpn.generate_anchors import generate_anchors
from utils.config import cfg
from utils.net_utils import bbox_overlaps, compute_targets


class AnchorTargetLayer(nn.Module):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def __init__(self):
        super(AnchorTargetLayer, self).__init__()
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.anchors = generate_anchors(ratios=np.array(cfg.ANCHOR_RATIOS),
                                        scales=np.array(cfg.ANCHOR_SCALES))
        self.num_anchors = self.anchors.shape[0]
        self.allowed_border = 0  # allow boxes to sit over the edge by a small amount

    def forward(self, rpn_cls_score, gt_boxes, im_info, use_rand=True):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert rpn_cls_score.size(0) == 1, 'Single batch only.'

        height, width = rpn_cls_score.shape[-2:]
        gt_boxes = gt_boxes.cpu().numpy()  # gt_boxes: (x1, y1, x2, y2, class, pid)
        im_info = im_info[0].cpu().numpy()

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self.feat_stride
        shift_y = np.arange(0, height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K * A, 4) shifted anchors
        A = self.num_anchors
        K = shifts.shape[0]
        anchors = self.anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where((anchors[:, 0] >= -self.allowed_border) &
                               (anchors[:, 1] >= -self.allowed_border) &
                               (anchors[:, 2] < im_info[1] + self.allowed_border) &  # width
                               (anchors[:, 3] < im_info[0] + self.allowed_border))[0]  # height

        # keep only inside anchors
        anchors = anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty(len(inds_inside), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps: (ex, gt)
        overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),
                                 np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            if use_rand:
                disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            else:
                disable_inds = fg_inds[:len(fg_inds) - num_fg]
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            if use_rand:
                disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            else:
                disable_inds = bg_inds[:len(bg_inds) - num_bg]
            labels[disable_inds] = -1

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert (cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1)
            positive_weights = cfg.TRAIN.RPN_POSITIVE_WEIGHT / np.sum(labels == 1)
            negative_weights = (1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / np.sum(labels == 0)
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        # map up to original set of anchors
        labels = unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))

        # bbox_targets
        bbox_targets = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width

        return (torch.from_numpy(labels).cuda(),
                torch.from_numpy(bbox_targets).cuda(),
                torch.from_numpy(bbox_inside_weights).cuda(),
                torch.from_numpy(bbox_outside_weights).cuda())


def unmap(data, count, inds, fill=0):
    """Unmap a subset of items (data) back to the original set of items (of size count)."""
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
