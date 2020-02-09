import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIPool, nms

from datasets.data_processing import img_preprocessing
from models.backbone import Backbone
from models.head import Head
from oim.labeled_matching_layer import LabeledMatchingLayer
from oim.unlabeled_matching_layer import UnlabeledMatchingLayer
from rpn.proposal_target_layer import ProposalTargetLayer
from rpn.rpn_layer import RPN
from utils.boxes import bbox_transform_inv, clip_boxes
from utils.config import cfg
from utils.utils import smooth_l1_loss


class Network(nn.Module):
    """
    Person search network.

    Paper: Joint Detection and Identification Feature Learning for Person Search
           Tong Xiao, Shuang Li, Bochao Wang, Liang Lin, Xiaogang Wang
    """

    def __init__(self):
        super(Network, self).__init__()
        rpn_depth = 1024  # Depth of the feature map fed into RPN
        num_classes = 2  # Background and foreground
        self.backbone = Backbone()
        self.head = Head()
        self.rpn = RPN(rpn_depth)
        self.roi_pool = RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.cls_score = nn.Linear(2048, num_classes)
        self.bbox_pred = nn.Linear(2048, num_classes * 4)
        self.feature = nn.Linear(2048, 256)
        self.proposal_target_layer = ProposalTargetLayer(num_classes)
        self.labeled_matching_layer = LabeledMatchingLayer()
        self.unlabeled_matching_layer = UnlabeledMatchingLayer()

        self.freeze_blocks()

    def forward(self, img, img_info, gt_boxes, probe_roi=None):
        """
        Args:
            img (Tensor): Single image data.
            img_info (Tensor): (height, width, scale)
            gt_boxes (Tensor): Ground-truth boxes in (x1, y1, x2, y2, class, person_id) format.
            probe_roi (Tensor): Take probe_roi as proposal instead of using RPN.

        Returns:
            proposals (Tensor): Region proposals produced by RPN in (0, x1, y1, x2, y2) format.
            probs (Tensor): Classification probability of these proposals.
            proposal_deltas (Tensor): Proposal regression deltas.
            features (Tensor): Extracted features of these proposals.
            rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox and loss_oim (Tensor): Training losses.
        """
        assert img.size(0) == 1, "Single batch only."

        # Extract basic feature from image data
        base_feat = self.backbone(img)

        if probe_roi is None:
            # Feed basic feature map to RPN to obtain rois
            proposals, rpn_loss_cls, rpn_loss_bbox = self.rpn(base_feat, img_info, gt_boxes)
        else:
            # Take given probe_roi as proposal if probe_roi is not None
            proposals, rpn_loss_cls, rpn_loss_bbox = probe_roi, 0, 0

        if self.training:
            # Sample some proposals and assign them ground-truth targets
            (
                proposals,
                cls_labels,
                pid_labels,
                gt_proposal_deltas,
                proposal_inside_ws,
                proposal_outside_ws,
            ) = self.proposal_target_layer(proposals, gt_boxes)
        else:
            cls_labels, pid_labels, gt_proposal_deltas, proposal_inside_ws, proposal_outside_ws = [
                None
            ] * 5

        # RoI pooling based on region proposals
        pooled_feat = self.roi_pool(base_feat, proposals)

        # Extract the features of proposals
        proposal_feat = self.head(pooled_feat).squeeze(2).squeeze(2)

        scores = self.cls_score(proposal_feat)
        probs = F.softmax(scores, dim=1)
        proposal_deltas = self.bbox_pred(proposal_feat)
        features = F.normalize(self.feature(proposal_feat))

        if self.training:
            loss_cls = F.cross_entropy(scores, cls_labels)
            loss_bbox = smooth_l1_loss(
                proposal_deltas, gt_proposal_deltas, proposal_inside_ws, proposal_outside_ws
            )

            # OIM loss
            labeled_matching_scores = self.labeled_matching_layer(features, pid_labels)
            labeled_matching_scores *= 10
            unlabeled_matching_scores = self.unlabeled_matching_layer(features, pid_labels)
            unlabeled_matching_scores *= 10
            matching_scores = torch.cat((labeled_matching_scores, unlabeled_matching_scores), dim=1)
            pid_labels = pid_labels.clone()
            pid_labels[pid_labels == -2] = -1
            loss_oim = F.cross_entropy(matching_scores, pid_labels, ignore_index=-1)
        else:
            loss_cls, loss_bbox, loss_oim = 0, 0, 0

        return (
            proposals,
            probs,
            proposal_deltas,
            features,
            rpn_loss_cls,
            rpn_loss_bbox,
            loss_cls,
            loss_bbox,
            loss_oim,
        )

    def freeze_blocks(self):
        """
        The reason why we freeze all BNs in the backbone: The batch size is 1
        in the backbone, so BN is not stable.

        Reference: https://github.com/ShuangLI59/person_search/issues/87
        """
        for p in self.backbone.SpatialConvolution_0.parameters():
            p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                for p in m.parameters():
                    p.requires_grad = False

        # Frozen all bn layers in backbone
        self.backbone.apply(set_bn_fix)

    def train(self, mode=True):
        """
        It's not enough to just freeze all BNs in backbone.
        Setting them to eval mode is also needed.
        """
        nn.Module.train(self, mode)

        if mode:
            # Set all bn layers in backbone to eval mode
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.eval()

            self.backbone.apply(set_bn_eval)

    def inference(self, img, probe_roi=None, threshold=0.75):
        """
        End to end inference. Specific behavior depends on probe_roi.
        If probe_roi is None, detect persons in the image and extract their features.
        Otherwise, extract the feature of the probe RoI in the image.

        Args:
            img (np.ndarray[H, W, C]): Image of BGR order.
            probe_roi (np.ndarray[4]): The RoI to be extracting feature.
            threshold (float): The threshold used to remove those bounding boxes with low scores.

        Returns:
            detections (Tensor[N, 5]): Detected person bounding boxes in
                                       (x1, y1, x2, y2, score) format.
            features (Tensor[N, 256]): Features of these bounding boxes.
        """
        device = self.cls_score.weight.device
        processed_img, scale = img_preprocessing(img)
        # [C, H, W] -> [N, C, H, W]
        processed_img = torch.from_numpy(processed_img).unsqueeze(0).to(device)
        # img_info: (height, width, scale)
        img_info = torch.Tensor([processed_img.shape[2], processed_img.shape[3], scale]).to(device)
        if probe_roi is not None:
            probe_roi = torch.from_numpy(probe_roi).float().view(1, 4)
            probe_roi *= scale
            # Add an extra 0, which means the probe_roi is from the first image in the batch
            probe_roi = torch.cat((torch.zeros(1, 1), probe_roi.float()), dim=1).to(device)

        with torch.no_grad():
            proposals, probs, proposal_deltas, features, _, _, _, _, _ = self.forward(
                processed_img, img_info, None, probe_roi
            )

        if probe_roi is not None:
            return features

        # Unscale proposals back to raw image space
        proposals = proposals[:, 1:5] / scale
        # Unnormalize proposal deltas
        num_classes = proposal_deltas.shape[1] // 4
        stds = torch.Tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(num_classes).to(device)
        means = torch.Tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(num_classes).to(device)
        proposal_deltas = proposal_deltas * stds + means
        # Apply proposal regression deltas
        boxes = bbox_transform_inv(proposals, proposal_deltas)
        boxes = clip_boxes(boxes, img.shape)

        # Remove those boxes with scores below the threshold
        j = 1  # Only consider foreground class
        keep = torch.nonzero(probs[:, j] > threshold)[:, 0]
        boxes = boxes[keep, j * 4 : (j + 1) * 4]
        probs = probs[keep, j]
        features = features[keep]

        # Remove redundant boxes with NMS
        detections = torch.cat((boxes, probs.unsqueeze(1)), dim=1)
        keep = nms(boxes, probs, cfg.TEST.NMS)
        detections = detections[keep]
        features = features[keep]

        return detections, features
