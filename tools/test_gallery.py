"""
Author: https://github.com/ShuangLI59/person_search.git
Description: Tools for testing gallery.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import nms
from tqdm import tqdm

from test_utils import get_image_blob
from utils.config import cfg
from utils.net_utils import bbox_transform_inv, clip_boxes


def im_detect(net, im):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        roidb (an roidb item): to provide gt_boxes if necessary
        blob_names (list of str): list of feature blob names to be extracted

    Returns:
        boxes (ndarray): R x (4 * K) array of predicted bounding boxes
        scores (ndarray): R x K array of object class scores (K includes
                          background as object category 0)
        features (dict of ndarray): {blob name: R x D array of features}
    """
    im_blob, im_scales = get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    blobs = {
        'data': im_blob,
        'im_info': np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32)
    }

    pid_prob, bbox_pred, feat, _, _, _, _, _ = net(torch.from_numpy(blobs['data']).cuda(),
                                                   torch.from_numpy(blobs['im_info']).cuda(), 0)
    pid_prob = pid_prob.detach().cpu().numpy()
    bbox_pred = bbox_pred.detach().cpu().numpy()
    feat = feat.detach().cpu().numpy()

    # unscale rois back to raw image space
    rois = net.rois.cpu().numpy()
    boxes = rois[:, 1:5] / im_scales[0]

    # the first column of the pid_prob is the non-person box score
    scores = pid_prob[:, 0]
    scores = scores[:, np.newaxis]
    scores = np.hstack([scores, 1. - scores])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        # As we no longer scale and shift the bbox_pred weights when snapshot,
        # we need to manually do this during test.
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS and cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            num_classes = box_deltas.shape[1] // 4
            stds = np.tile(cfg.TRAIN.BBOX_NORMALIZE_STDS, num_classes)
            means = np.tile(cfg.TRAIN.BBOX_NORMALIZE_MEANS, num_classes)
            box_deltas = box_deltas * stds + means
        boxes = bbox_transform_inv(boxes, box_deltas)
        boxes = clip_boxes(boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        boxes = np.tile(boxes, (1, scores.shape[1]))

    features = {'feat' : feat.copy()}

    return boxes, scores, features


def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    im = im[:, :, (2, 1, 0)]
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                              bbox[2] - bbox[0],
                                              bbox[3] - bbox[1],
                                              fill=False,
                                              edgecolor='g',
                                              linewidth=3))
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()


def detect_and_exfeat(net, imdb, thresh=0.05, vis=False):
    assert imdb.num_classes == 2, "Only support two-class detection"

    num_images = imdb.num_images

    # all detections are collected into:
    #    all_boxes[image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    #    all_features[blob][image] = N x D array of features
    all_boxes = [0 for _ in range(num_images)]
    all_features = {'feat': [0 for _ in range(num_images)]}

    for i in tqdm(range(num_images)):
        im = cv2.imread(imdb.image_path_at(i))
        # roidb = imdb.roidb[i]

        boxes, scores, feat_dic = im_detect(net, im)

        j = 1  # only consider j = 1 (foreground class)
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j * 4:(j + 1) * 4].astype(np.float32)
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(torch.from_numpy(cls_boxes).cuda(),
                   torch.from_numpy(cls_scores).cuda(),
                   cfg.TEST.NMS).cpu().numpy()
        all_boxes[i] = cls_dets[keep]
        # for blob, feat in feat_dic.iteritems():
        #     all_features[blob][i] = feat[inds][keep]
        all_features['feat'][i] = feat_dic['feat'][inds][keep]

        if vis:
            vis_detections(im, imdb.classes[j], all_boxes[i])

    return all_boxes, all_features


# def usegt_and_exfeat(net, imdb, start=None, end=None, blob_names=None):
#     start = start or 0
#     end = end or imdb.num_images
#     num_images = end - start

#     # all detections are collected into:
#     #    all_boxes[image] = N x 5 array of detections (gt) in
#     #    (x1, y1, x2, y2, score)
#     #    all_features[blob][image] = N x D array of features
#     all_boxes = [0 for _ in range(num_images)]
#     all_features = {} if blob_names is None else {
#         blob: [0 for _ in range(num_images)] for blob in blob_names}

#     # timers
#     _t = {'gt_exfeat': Timer(), 'misc': Timer()}

#     for i in range(num_images):
#         im = cv2.imread(imdb.image_path_at(start + i))
#         gt = imdb.roidb[start + i]['boxes']

#         _t['gt_exfeat'].tic()
#         feat_dic = im_exfeat(net, im, gt, blob_names)
#         _t['gt_exfeat'].toc()

#         all_boxes[i] = np.hstack((gt, np.ones((gt.shape[0], 1)))).astype(np.float32)
#         for blob, feat in feat_dic.iteritems():
#             all_features[blob][i] = feat

#         print('gt_exfeat: %s/%s %ss' % (i + 1, num_images, _t['gt_exfeat'].average_time))

#     return all_boxes, all_features


def demo_detect(net, filename, blob_name='feat', threshold=0.5):
    """Detect persons in a gallery image and extract their features

    Arguments:
        net (caffe.Net): trained network
        filename (str): path to a gallery image file (jpg or png)
        blob_name (str): feature blob name. Default 'feat'
        threshold (float): detection score threshold. Default 0.5

    Returns:
        boxes (ndarray): N x 5 detected boxes in format [x1, y1, x2, y2, score]
        features (ndarray): N x D features matrix
    """
    im = cv2.imread(filename)
    boxes, scores, feat_dic = im_detect(net, im)

    j = 1  # only consider j = 1 (foreground class)
    inds = np.where(scores[:, j] > threshold)[0]
    cls_scores = scores[inds, j]
    cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
    boxes = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(torch.from_numpy(cls_boxes),
               torch.from_numpy(cls_scores),
               cfg.TEST.NMS)

    boxes = boxes[keep]
    features = feat_dic[blob_name][inds][keep]

    if boxes.shape[0] == 0:
        return None, None

    features = features.reshape(features.shape[0], -1)
    return boxes, features
