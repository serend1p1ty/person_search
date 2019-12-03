from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector


@DETECTORS.register_module()
class PersonSearchDetector(TwoStageDetector):
    def __init__(
        self, backbone, rpn_head, roi_head, train_cfg, test_cfg, neck=None, pretrained=None
    ):
        super(PersonSearchDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_pids,
        gt_bboxes_ignore=None,
        gt_masks=None,
        proposals=None,
        **kwargs
    ):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas, cfg=proposal_cfg)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_pids,
            gt_bboxes_ignore,
            gt_masks,
            **kwargs
        )
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale, flag=(proposals is not None)
        )
