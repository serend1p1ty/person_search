import cv2
import numpy as np
import torch

from utils.config import cfg


def build_net_input(roidb):
    """
    Given a roidb, build the input of the network.

    Args:
        roidb: A dictionary saving the annotation information.

    Returns:
        processed_img (Tensor[C, H, W]): Processed image.
        img_info (Tensor[3]): (height, width, scale)
        gt_boxes (Tensor[N, 6]): Ground-truth boxes in (x1, y1, x2, y2, class, person_id) format.
    """
    img = cv2.imread(roidb["image"])
    processed_img, scale = img_preprocessing(img, roidb["flipped"])

    # GT boxes: (x1, y1, x2, y2, class, person_id)
    gt_boxes = np.empty((roidb["gt_boxes"].shape[0], 6), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb["gt_boxes"] * scale
    gt_boxes[:, 4] = 1  # 0: background, 1: person
    gt_boxes[:, 5] = roidb["gt_pids"]

    img_info = torch.Tensor([processed_img.shape[1], processed_img.shape[2], scale])

    return torch.from_numpy(processed_img), img_info, torch.from_numpy(gt_boxes)


def img_preprocessing(img, flipped=False):
    """
    Image preprocessing: flip (optional), subtract mean, scale.

    Args:
        img (np.ndarray[H, W, C]): Origin image in BGR order.
        flipped (bool): Whether to flip the image.

    Returns:
        processed_img (np.ndarray[C, H, W]): Processed image.
        scale (float): The scale relative to the original image.
    """
    if flipped:
        img = img[:, ::-1, :]

    processed_img = img.astype(np.float32)
    processed_img -= cfg.PIXEL_MEANS

    img_size_min = np.min(processed_img.shape[0:2])
    img_size_max = np.max(processed_img.shape[0:2])
    scale = float(cfg.SCALE) / float(img_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(scale * img_size_max) > cfg.MAX_SIZE:
        scale = float(cfg.MAX_SIZE) / float(img_size_max)
    processed_img = cv2.resize(
        processed_img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
    )

    processed_img = processed_img.transpose((2, 0, 1))  # [H, W, C] -> [C, H, W]
    return processed_img, scale
