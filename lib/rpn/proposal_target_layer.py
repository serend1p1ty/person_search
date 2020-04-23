# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import torch
import torch.nn as nn

from utils.boxes import bbox_overlaps, bbox_transform
from utils.config import cfg
from utils.utils import torch_rand_choice


class ProposalTargetLayer(nn.Module):
    """
    Sample some proposals at the specified positive and negative ratio.

    And assign ground-truth targets (cls_labels, pid_labels, deltas, inside_weights,
    outside_weights) to these sampled proposals.

    BTW:
    pid_label = -1 -----> foreground proposals containing an unlabeled person.
    pid_label = -2 -----> background proposals.
    """

    def __init__(self, num_classes, bg_pid_label=-2):
        super(ProposalTargetLayer, self).__init__()
        self.num_classes = num_classes
        self.bg_pid_label = bg_pid_label

    def forward(self, proposals, gt_boxes):
        """
        Args:
            proposals (Tensor): Region proposals in (0, x1, y1, x2, y2) format coming from RPN.
            gt_boxes (Tensor): Ground-truth boxes in (x1, y1, x2, y2, class, person_id) format.

        Returns:
            proposals (Tensor[N, 5]): Sampled proposals.
            cls_labels (Tensor[N]): Ground-truth classification labels of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.
            deltas (Tensor[N, num_classes * 4]):  Ground-truth regression deltas of the proposals.
            inside_weights, outside_weights (Tensor): Used to calculate smooth_l1_loss.
        """
        assert torch.all(proposals[:, 0] == 0), "Single batch only."

        # Include ground-truth boxes in the set of candidate proposals
        zeros = gt_boxes.new(gt_boxes.shape[0], 1).zero_()
        proposals = torch.cat((proposals, torch.cat((zeros, gt_boxes[:, :4]), dim=1)), dim=0)

        overlaps = bbox_overlaps(proposals[:, 1:5], gt_boxes[:, :4])
        max_overlaps, argmax_overlaps = overlaps.max(dim=1)
        cls_labels = gt_boxes[argmax_overlaps, 4]
        pid_labels = gt_boxes[argmax_overlaps, 5]

        # Sample some proposals at the specified positive and negative ratio
        batch_size = cfg.TRAIN.BATCH_SIZE
        num_fg = round(cfg.TRAIN.FG_FRACTION * batch_size)

        # Sample foreground proposals
        fg_inds = torch.nonzero(max_overlaps >= cfg.TRAIN.FG_THRESH)[:, 0]
        num_fg = min(num_fg, fg_inds.numel())
        if fg_inds.numel() > 0:
            fg_inds = torch_rand_choice(fg_inds, num_fg)

        # Sample background proposals
        bg_inds = torch.nonzero(
            (max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps >= cfg.TRAIN.BG_THRESH_LO)
        )[:, 0]
        num_bg = min(batch_size - num_fg, bg_inds.numel())
        if bg_inds.numel() > 0:
            bg_inds = torch_rand_choice(bg_inds, num_bg)

        # assert num_fg + num_bg == batch_size

        keep = torch.cat((fg_inds, bg_inds))
        cls_labels = cls_labels[keep]
        pid_labels = pid_labels[keep]
        proposals = proposals[keep]

        # Correct the cls_labels and pid_labels of bg proposals
        cls_labels[num_fg:] = 0
        pid_labels[num_fg:] = self.bg_pid_label

        deltas, inside_weights, outside_weights = self.get_regression_targets(
            proposals[:, 1:5], gt_boxes[argmax_overlaps][keep, :4], cls_labels, self.num_classes,
        )

        return (
            proposals,
            cls_labels.long(),
            pid_labels.long(),
            deltas,
            inside_weights,
            outside_weights,
        )

    @staticmethod
    def get_regression_targets(proposals, gt_boxes, cls_labels, num_classes):
        """
        Args:
            proposals (Tensor): Sampled proposals in (x1, y1, x2, y2) format.
            gt_boxes (Tensor): Ground-truth boxes in (x1, y1, x2, y2) format.
            cls_labels (Tensor): Classification labels of the proposals.
            num_classes (int): Number of classes.

        Returns:
            deltas ([N, num_classes * 4]): Proposal regression deltas.
            inside_weights, outside_weights (Tensor): Used to calculate smooth_l1_loss.
        """
        deltas_data = bbox_transform(proposals, gt_boxes)
        # Normalize targets by a precomputed mean and stdev
        means = gt_boxes.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        stds = gt_boxes.new(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        deltas_data = (deltas_data - means) / stds

        deltas = gt_boxes.new(proposals.size(0), 4 * num_classes).zero_()
        inside_weights = deltas.clone()
        outside_weights = deltas.clone()
        fg_inds = torch.nonzero(cls_labels > 0)[:, 0]
        for ind in fg_inds:
            cls = int(cls_labels[ind])
            start = 4 * cls
            end = start + 4
            deltas[ind, start:end] = deltas_data[ind]
            inside_weights[ind, start:end] = gt_boxes.new(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)
            outside_weights[ind, start:end] = gt_boxes.new(4).fill_(1)
        return deltas, inside_weights, outside_weights
