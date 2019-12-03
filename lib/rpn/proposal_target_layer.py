"""
Author: Ross Girshick and Sean Bell
Description: Assign labels and regression targets to region proposals.
"""

import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn

from utils.config import cfg
from utils.net_utils import bbox_overlaps, bbox_transform


class ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, num_classes, bg_pid_label=5532):
        super(ProposalTargetLayer, self).__init__()
        self.num_classes = num_classes
        self.bg_pid_label = bg_pid_label

    def forward(self, all_rois, gt_boxes, use_rand=True):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = all_rois.cpu().numpy()

        # GT boxes (x1, y1, x2, y2, class, pid)
        gt_boxes = gt_boxes.cpu().numpy()

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :4]))))

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Single batch only.'

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

        # Sample rois with classification labels and bounding box regression targets
        sample_data = sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image,
                                  self.num_classes, self.bg_pid_label, use_rand)

        labels, rois, bbox_targets, bbox_inside_weights, aux_label = sample_data
        bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

        return (torch.from_numpy(rois).cuda(),
                torch.from_numpy(labels).long().cuda(),
                torch.from_numpy(bbox_targets).cuda(),
                torch.from_numpy(bbox_inside_weights).cuda(),
                torch.from_numpy(bbox_outside_weights).cuda(),
                torch.from_numpy(aux_label).long().cuda())


def get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(round(clss[ind]))
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                   / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, bg_pid_label, use_rand):
    """Generate a random sample of RoIs comprising foreground and background examples."""
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
                             np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        if use_rand:
            fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
        else:
            fg_inds = fg_inds[:fg_rois_per_this_image]

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    # Guard against the case when an image has fewer than bg_rois_per_this_image background RoIs
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        if use_rand:
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
        else:
            bg_inds = bg_inds[:bg_rois_per_this_image]

    assert fg_rois_per_this_image + bg_rois_per_this_image == rois_per_image, \
        "fg_boxes + bg_boxes must be %s" % rois_per_image

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    # Auxiliary label if available
    pid_label = None
    if gt_boxes.shape[1] > 5:
        pid_label = gt_boxes[gt_assignment, 5]
        pid_label = pid_label[keep_inds]
        pid_label[fg_rois_per_this_image:] = bg_pid_label

    bbox_target_data = compute_targets(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights, pid_label
