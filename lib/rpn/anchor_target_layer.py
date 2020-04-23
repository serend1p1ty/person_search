# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import torch
import torch.nn as nn

from rpn.generate_anchors import generate_anchors
from utils.boxes import bbox_overlaps, bbox_transform
from utils.config import cfg
from utils.utils import torch_rand_choice


class AnchorTargetLayer(nn.Module):
    """
    Assign ground-truth targets (labels, deltas, inside_weights, outside_weights) to anchors.
    """

    def __init__(self):
        super(AnchorTargetLayer, self).__init__()
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.anchors = generate_anchors()
        self.num_anchors = self.anchors.size(0)

    def forward(self, scores, gt_boxes, img_info):
        """
        Args:
            scores (Tensor[1, num_anchors * num_classes, H, W]): Classification scores.
            gt_boxes (Tensor[N, 6]): Ground-truth boxes in (x1, y1, x2, y2, class, person_id) format
            img_info (Tensor[3]): (height, width, scale)

        Returns:
            labels (Tensor): Ground-truth labels of the anchors.
            deltas (Tensor): Ground-truth regression deltas of the anchors.
            inside_weights, outside_weights (Tensor): Used to calculate smooth_l1_loss
        """
        # Algorithm:
        #
        # For each (H, W) location i
        #     Generate A anchors centered on cell i
        # Filter out-of-image anchors
        # Measure the overlaps between anchors and gt_boxes
        # Assign labels, deltas, inside_weights, outside_weights for each anchor

        assert scores.size(0) == 1, "Single batch only."
        height, width = scores.shape[-2:]

        # Enumerate all shifts (NOTE: torch.meshgrid is different from np.meshgrid)
        shift_x = torch.arange(0, width) * self.feat_stride
        shift_y = torch.arange(0, height) * self.feat_stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x, shift_y = shift_x.contiguous(), shift_y.contiguous()
        shifts = torch.stack(
            (shift_x.view(-1), shift_y.view(-1), shift_x.view(-1), shift_y.view(-1)), dim=1
        )
        shifts = shifts.type_as(gt_boxes)

        # Enumerate all shifted anchors:
        # Add A anchors (1, A, 4) to K shifts (K, 1, 4) to get shifted anchors (K, A, 4)
        # Reshape to (K * A, 4) shifted anchors
        A = self.num_anchors
        K = shifts.size(0)
        self.anchors = self.anchors.type_as(gt_boxes)
        anchors = self.anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2)
        anchors = anchors.view(K * A, 4)

        # Only keep anchors inside the image
        keep = torch.nonzero(
            (anchors[:, 0] >= 0)
            & (anchors[:, 1] >= 0)
            & (anchors[:, 2] < img_info[1])
            & (anchors[:, 3] < img_info[0])
        )[:, 0]
        anchors = anchors[keep]

        overlaps = bbox_overlaps(anchors, gt_boxes[:, :4])
        max_overlaps, argmax_overlaps = overlaps.max(dim=1)
        gt_max_overlaps = overlaps.max(dim=0)[0]
        gt_argmax_overlaps = torch.nonzero(overlaps == gt_max_overlaps)[:, 0]

        # label: 1 is positive, 0 is negative, -1 is dont care
        # The anchors which satisfied both positive and negative conditions will be as positive
        labels = gt_boxes.new(len(keep)).fill_(-1)
        # bg labels: below threshold IOU
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1
        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        # Subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = torch.nonzero(labels == 1)[:, 0]
        if len(fg_inds) > num_fg:
            disable_inds = torch_rand_choice(fg_inds, len(fg_inds) - num_fg)
            labels[disable_inds] = -1

        # Subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum(labels == 1)
        bg_inds = torch.nonzero(labels == 0)[:, 0]
        if len(bg_inds) > num_bg:
            disable_inds = torch_rand_choice(bg_inds, len(bg_inds) - num_bg)
            labels[disable_inds] = -1

        deltas = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

        inside_weights = gt_boxes.new(deltas.shape).zero_()
        inside_weights[labels == 1] = gt_boxes.new(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        outside_weights = gt_boxes.new(deltas.shape).zero_()
        num_examples = torch.sum(labels >= 0)
        outside_weights[labels == 1] = gt_boxes.new(1, 4).fill_(1) / num_examples
        outside_weights[labels == 0] = gt_boxes.new(1, 4).fill_(1) / num_examples

        def map2origin(data, count=K * A, inds=keep, fill=0):
            """Map to original set."""
            shape = (count,) + data.shape[1:]
            origin = torch.empty(shape).fill_(fill).type_as(gt_boxes)
            origin[inds] = data
            return origin

        labels = map2origin(labels, fill=-1)
        deltas = map2origin(deltas)
        inside_weights = map2origin(inside_weights)
        outside_weights = map2origin(outside_weights)

        labels = labels.view(1, height, width, A).permute(0, 3, 1, 2)
        labels = labels.contiguous().view(1, 1, A * height, width)
        deltas = deltas.view(1, height, width, A * 4).permute(0, 3, 1, 2)
        inside_weights = inside_weights.view(1, height, width, A * 4).permute(0, 3, 1, 2)
        outside_weights = outside_weights.view(1, height, width, A * 4).permute(0, 3, 1, 2)

        return labels, deltas, inside_weights, outside_weights
