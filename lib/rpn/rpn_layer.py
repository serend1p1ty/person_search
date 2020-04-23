import torch.nn as nn
import torch.nn.functional as F

from rpn.anchor_target_layer import AnchorTargetLayer
from rpn.proposal_layer import ProposalLayer
from utils.config import cfg
from utils.utils import smooth_l1_loss


class RPN(nn.Module):
    """
    Region proposal network.
    """

    def __init__(self, input_depth):
        super(RPN, self).__init__()
        self.num_anchors = len(cfg.ANCHOR_SCALES) * len(cfg.ANCHOR_RATIOS)
        # 3x3 conv for the hidden representation
        self.rpn_conv = nn.Conv2d(input_depth, 512, 3, 1, 1)
        # 1x1 conv for predicting bg/fg classification score
        # Output channel: 9(anchors) * 2(bg/fg)
        self.rpn_cls_score = nn.Conv2d(512, self.num_anchors * 2, 1, 1, 0)
        # 1x1 conv for predicting anchor box offset
        # Output channel: 9(anchors) * 4(coords)
        self.rpn_bbox_pred = nn.Conv2d(512, self.num_anchors * 4, 1, 1, 0)
        self.rpn_proposal = ProposalLayer()
        self.rpn_anchor_target = AnchorTargetLayer()

    @staticmethod
    def reshape(x, d):
        x = x.view(x.size(0), d, -1, x.size(3))
        return x

    def forward(self, base_feat, img_info, gt_boxes):
        """
        Args:
            base_feat (Tensor[1, C, H, W]): Basic feature extracted by backbone.
            img_info (Tensor[3]): (height, width, scale)
            gt_boxes (Tensor[N, 6]): Ground-truth boxes in (x1, y1, x2, y2, class, person_id) format

        Returns:
            proposals (Tensor[N, 5]): Predicted region proposals in (0, x1, y1, x2, y2) format.
            rpn_loss_cls, rpn_loss_bbox (Tensor): Training losses.
        """
        assert base_feat.size(0) == 1, "Single batch only."

        rpn_conv = F.relu(self.rpn_conv(base_feat), inplace=True)

        # Predict classification score
        scores = self.rpn_cls_score(rpn_conv)
        scores_reshape = self.reshape(scores, 2)
        probs_reshape = F.softmax(scores_reshape, 1)
        probs = self.reshape(probs_reshape, self.num_anchors * 2)

        # Predict anchor regression deltas
        anchor_deltas = self.rpn_bbox_pred(rpn_conv)

        # Produce region proposals
        proposals = self.rpn_proposal(probs.data, anchor_deltas.data, img_info)

        rpn_loss_cls = 0
        rpn_loss_bbox = 0

        if self.training:
            assert gt_boxes is not None
            anchor_target = self.rpn_anchor_target(scores.data, gt_boxes, img_info)

            # Classification loss
            scores = scores_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
            gt_anchor_labels = anchor_target[0].view(-1).long()
            rpn_loss_cls = F.cross_entropy(scores, gt_anchor_labels, ignore_index=-1)

            # Anchor regression loss
            gt_anchor_deltas, anchor_inside_ws, anchor_outside_ws = anchor_target[1:]
            rpn_loss_bbox = smooth_l1_loss(
                anchor_deltas, gt_anchor_deltas, anchor_inside_ws, anchor_outside_ws, sigma=3
            )

        return proposals, rpn_loss_cls, rpn_loss_bbox
