import torch
import torch.nn as nn
import torch.nn.functional as F

from bbox_nms import multiclass_nms
from mmdet.core import auto_fp16, force_fp32, multi_apply
from mmdet.models.builder import HEADS
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from oim_loss import OIMLoss


@HEADS.register_module()
class PersonSearchBBoxHead(BBoxHead):
    def __init__(self, *args, **kwargs):
        super(PersonSearchBBoxHead, self).__init__(*args, **kwargs)
        self.fc_feat = nn.Linear(self.in_channels, 256)
        self.loss_oim = OIMLoss()

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        feature = F.normalize(self.fc_feat(x))
        return cls_score, bbox_pred, feature

    def _get_target_single(
        self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, pos_gt_pids, cfg
    ):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        bg_pid = -2  # person ID of background proposal
        pids = pos_bboxes.new_full((num_samples,), bg_pid, dtype=torch.long)
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,), self.num_classes, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            pids[:num_pos] = pos_gt_pids
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return pids, labels, label_weights, bbox_targets, bbox_weights

    def get_targets(
        self, sampling_results, gt_bboxes, gt_labels, gt_pids, rcnn_train_cfg, concat=True
    ):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_gt_pids_list = []
        for pids, res in zip(gt_pids, sampling_results):
            pos_gt_pids_list.append(pids[res.pos_assigned_gt_inds])
        pids, labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_gt_pids_list,
            cfg=rcnn_train_cfg,
        )

        if concat:
            pids = torch.cat(pids, 0)
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return pids, labels, label_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=("cls_score", "bbox_pred"))
    def loss(
        self,
        cls_score,
        bbox_pred,
        feature,
        rois,
        pids,
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        reduction_override=None,
    ):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
            if cls_score.numel() > 0:
                losses["loss_cls"] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override,
                )
                losses["acc"] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[
                        pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]
                    ]
                losses["loss_bbox"] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override,
                )
            else:
                losses["loss_bbox"] = bbox_pred.sum() * 0
        losses["loss_oim"] = self.loss_oim(feature, pids)
        return losses

    @force_fp32(apply_to=("cls_score", "bbox_pred"))
    def get_bboxes(
        self, rois, cls_score, bbox_pred, feature, img_shape, scale_factor, rescale=False, cfg=None
    ):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                    bboxes.size()[0], -1
                )

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels, det_features = multiclass_nms(
                bboxes, scores, feature, cfg.score_thr, cfg.nms, cfg.max_per_img
            )

            return det_bboxes, det_labels, det_features
