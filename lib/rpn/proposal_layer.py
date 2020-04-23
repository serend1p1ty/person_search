# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import torch
import torch.nn as nn
from torchvision.ops import nms

from rpn.generate_anchors import generate_anchors
from utils.boxes import bbox_transform_inv, clip_boxes
from utils.config import cfg


class ProposalLayer(nn.Module):
    """
    Outputs proposals by applying estimated regression deltas to a set of anchors.
    """

    def __init__(self):
        super(ProposalLayer, self).__init__()
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.anchors = generate_anchors()
        self.num_anchors = self.anchors.size(0)

    def forward(self, probs, anchor_deltas, img_info):
        """
        Args:
            probs (Tensor): Classification probability of the anchors.
            anchor_deltas (Tensor): Anchor regression deltas.
            img_info (Tensor[3]): (height, width, scale)

        Returns:
            proposals (Tensor[N, 5]): Predicted region proposals in (0, x1, y1, x2, y2) format.
                                      0 means these proposals are from the first image in the batch.
        """
        # Algorithm:
        #
        # For each (H, W) location i:
        #     Generate A anchors centered on cell i
        #     Apply predicted anchor regression deltas at cell i to each of the A anchors
        # Clip predicted boxes to image
        # Remove predicted boxes with either height or width < threshold
        # Sort all (proposal, score) pairs by score from highest to lowest
        # Take top pre_nms_topN proposals before NMS
        # Apply NMS with threshold 0.7 to remaining proposals
        # Take after_nms_topN proposals after NMS

        assert probs.size(0) == 1, "Single batch only."

        cfg_key = "TRAIN" if self.training else "TEST"
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE

        # The first set of num_anchors channels are bg probs
        # The second set are the fg probs, which we want
        probs = probs[:, self.num_anchors :, :, :]

        # 1. Generate proposals from regression deltas and shifted anchors
        height, width = probs.shape[-2:]

        # Enumerate all shifts (NOTE: torch.meshgrid is different from np.meshgrid)
        shift_x = torch.arange(0, width) * self.feat_stride
        shift_y = torch.arange(0, height) * self.feat_stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x, shift_y = shift_x.contiguous(), shift_y.contiguous()
        shifts = torch.stack(
            (shift_x.view(-1), shift_y.view(-1), shift_x.view(-1), shift_y.view(-1)), dim=1
        )
        shifts = shifts.type_as(probs)

        # Enumerate all shifted anchors:
        # Add A anchors (1, A, 4) to K shifts (K, 1, 4) to get shifted anchors (K, A, 4)
        # Reshape to (K * A, 4) shifted anchors
        A = self.num_anchors
        K = shifts.size(0)
        self.anchors = self.anchors.type_as(probs)
        anchors = self.anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2)
        anchors = anchors.view(K * A, 4)

        # Permute and reshape predicted anchor regression deltas to the same order as the anchors:
        # Anchor deltas will be (1, 4 * A, H, W) format
        # Permute to (1, H, W, 4 * A)
        # Reshape to (1 * H * W * A, 4)
        anchor_deltas = anchor_deltas.permute(0, 2, 3, 1).contiguous().view(-1, 4)

        # Safe-guard for unexpected large dw or dh value.
        # Since our proposals are only human, some background region features will never
        # receive gradients from bbox regression. Thus their predictions may drift away.
        anchor_deltas[:, 2:].clamp_(-10, 10)

        # Same story for the scores:
        # Scores are (1, A, H, W) format
        # Permute to (1, H, W, A)
        # Reshape to (1 * H * W * A, 1)
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, 1)

        # Convert anchors into proposals via regression deltas
        proposals = bbox_transform_inv(anchors, anchor_deltas)

        # 2. Clip predicted proposals to image
        proposals = clip_boxes(proposals, img_info[:2])

        # 3. Remove predicted boxes with either height or width < threshold
        # (NOTE: need to scale min_size with the input image scale stored in img_info[2])
        widths = proposals[:, 2] - proposals[:, 0] + 1
        heights = proposals[:, 3] - proposals[:, 1] + 1
        min_size = min_size * img_info[2]
        keep = torch.nonzero((widths >= min_size) & (heights >= min_size))[:, 0]
        proposals = proposals[keep]
        probs = probs[keep]

        # 4. Sort all (proposal, score) pairs by score from highest to lowest
        # 5. Take top pre_nms_topN (e.g. 6000)
        order = probs.view(-1).argsort(descending=True)
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order]
        probs = probs[order]

        # 6. Apply nms (e.g. threshold = 0.7)
        # 7. Take after_nms_topN (e.g. 300)
        # 8. Return the top proposals
        keep = nms(proposals, probs.squeeze(1), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep]
        probs = probs[keep]

        # proposals: [img_id, x1, y1, x2, y2]
        # Our RPN implementation only supports a single input image, so all img_ids are 0.
        proposals = torch.cat((torch.zeros(proposals.size(0), 1).type_as(probs), proposals), dim=1)
        return proposals
