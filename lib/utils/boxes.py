import torch


def bbox_transform(boxes, gt_boxes):
    """
    Compute regression deltas of transforming boxes to gt_boxes.

    Args:
        boxes (Tensor[N, 4])
        gt_boxes (Tensor[N, 4])

    Returns:
        deltas (Tensor[N, 4])
    """
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    dx = (gt_ctr_x - ctr_x) / widths
    dy = (gt_ctr_y - ctr_y) / heights
    dw = torch.log(gt_widths / widths)
    dh = torch.log(gt_heights / heights)

    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    return deltas


def bbox_transform_inv(boxes, deltas):
    """
    Apply regression deltas on the boxes.

    Args:
        boxes (Tensor[N, 4])
        deltas (Tensor[N, 4])

    Returns:
        pred_boxes (Tensor[N, 4])
    """
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = deltas.clone()
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1  # x2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1  # y2

    return pred_boxes


def clip_boxes(boxes, img_shape):
    """
    Clip boxes to image boundaries.

    Args:
        boxes (Tensor[N, 4])
        img_shape (Tensor[2]): (height, width)

    Returns:
        boxes (Tensor[N, 4]): Clipped boxes.
    """
    boxes[:, 0::4].clamp_(0, img_shape[1] - 1)
    boxes[:, 1::4].clamp_(0, img_shape[0] - 1)
    boxes[:, 2::4].clamp_(0, img_shape[1] - 1)
    boxes[:, 3::4].clamp_(0, img_shape[0] - 1)
    return boxes


def bbox_overlaps(boxes1, boxes2):
    """
    Compute the overlaps between boxes1 and boxes2.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        overlaps (Tensor[N, M])
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)

    iw = (
        torch.min(boxes1[:, 2:3], boxes2[:, 2:3].t())
        - torch.max(boxes1[:, 0:1], boxes2[:, 0:1].t())
        + 1
    ).clamp(min=0)
    ih = (
        torch.min(boxes1[:, 3:4], boxes2[:, 3:4].t())
        - torch.max(boxes1[:, 1:2], boxes2[:, 1:2].t())
        + 1
    ).clamp(min=0)
    ua = area1.view(-1, 1) + area2.view(1, -1) - iw * ih
    overlaps = iw * ih / ua

    return overlaps
