"""
Author: Ross Girshick and Sean Bell
Description: Get region proposals from anchors.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms

from rpn.generate_anchors import generate_anchors
from utils.config import cfg
from utils.net_utils import bbox_transform_inv, clip_boxes, filter_boxes


class ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self):
        super(ProposalLayer, self).__init__()
        self._feat_stride = cfg.FEAT_STRIDE[0]
        self._anchors = generate_anchors(ratios=np.array(cfg.ANCHOR_RATIOS),
                                         scales=np.array(cfg.ANCHOR_SCALES))
        self._num_anchors = self._anchors.shape[0]

    def forward(self, scores, bbox_deltas, im_info, use_nms=True):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert scores.size(0) == 1, 'Single batch only.'

        cfg_key = 'TRAIN' if self.training else 'TEST'
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = scores[:, self._num_anchors:, :, :].cpu().numpy()
        bbox_deltas = bbox_deltas.cpu().numpy()
        im_info = im_info[0].cpu().numpy()

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K * A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (H, W, A)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Safe-guard for unexpected large dw or dh value
        # Since our proposals are only human, some background region features
        # will never receive gradients from bbox regression. Thus their
        # predictions may drift away.
        bbox_deltas[:, 2:] = np.maximum(np.minimum(bbox_deltas[:, 2:], 10), -10)

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (H, W, A)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        if use_nms:
            keep = nms(torch.from_numpy(proposals), torch.from_numpy(scores).squeeze(1), nms_thresh)
        else:
            keep = np.arange(proposals.shape[0])
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all batch inds are 0
        blob = np.zeros((proposals.shape[0], 5), dtype=np.float32)
        blob[:, 1:] = proposals
        return torch.from_numpy(blob).cuda()
