"""
Author: Ross Girshick
Last editor: 520Chris
Description: Get minibatch blobs from given roidb.
"""

import cv2
import numpy as np
import numpy.random as npr

from utils.config import cfg


def get_minibatch(roidb):
    """Given a roidb, construct a minibatch."""
    num_images = len(roidb)
    assert num_images == 1, "Single batch only."

    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)

    # Get the input image blob
    im_blob, im_scales = get_image_blob(roidb, random_scale_inds)

    # GT boxes: (x1, y1, x2, y2, class, pid)
    gt_boxes = np.empty((roidb[0]['boxes'].shape[0], 6), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]["boxes"] * im_scales[0]
    gt_boxes[:, 4] = 1
    gt_boxes[:, 5] = roidb[0]["gt_pids"]

    blobs = {
        "data": im_blob,
        "gt_boxes": gt_boxes,
        "im_info": np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32)
    }

    return blobs


def get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified scales."""
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]["image"])
        if roidb[i]["flipped"]:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales


def im_list_to_blob(ims):
    """
    Convert a list of images into a network input. Assumes images
    are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)

    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    return im, im_scale
