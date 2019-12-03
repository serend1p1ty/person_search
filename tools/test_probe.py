"""
Author: https://github.com/ShuangLI59/person_search.git
Description: Tools fro testing probes.
"""

import torch
import cv2
import numpy as np
from tqdm import tqdm

from test_utils import get_image_blob, get_rois_blob


def im_exfeat(net, im, roi):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        roi (ndarray): 1 x 4 array of the target roi
        blob_names (list of str): list of feature blob names to be extracted

    Returns:
        features (dict of ndarray): {blob name: R x D array of features}
    """
    im_blob, im_scales = get_image_blob(im)

    blobs = {
        'data': im_blob,
        'im_info': np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32),
        'rois': get_rois_blob(roi, im_scales),
    }

    _, _, feat, _, _, _, _, _ = net(torch.from_numpy(blobs['data']).cuda(),
                                    torch.from_numpy(blobs['im_info']).cuda(),
                                    0, is_prob=True, rois=torch.from_numpy(blobs['rois']).cuda())
    feat = feat.detach().cpu().numpy()

    features = {'feat': feat.copy()}

    return features


def exfeat(net, probes):
    num_images = len(probes)

    # all_features[blob] = num_images x D array of features
    all_features = {'feat': [0 for _ in range(num_images)]}

    for i in tqdm(range(num_images)):
        im_name, roi = probes[i]
        im = cv2.imread(im_name)
        roi = roi.reshape(1, 4)

        feat_dic = im_exfeat(net, im, roi)

        for blob, feat in feat_dic.items():
            assert feat.shape[0] == 1
            all_features[blob][i] = feat[0]

    return all_features


def demo_exfeat(net, filename, roi, blob_name='feat'):
    """Extract feature for a probe person ROI in an image.

    Arguments:
        net (caffe.Net): trained network
        filename (str): path to a probe image file (jpg or png)
        roi (list or ndarray): target roi in format [x1, y1, x2, y2, score]
        blob_name (str): feature blob name. Default 'feat'

    Returns:
        feature (ndarray): D-dimensional vector
    """
    im = cv2.imread(filename)
    roi = np.asarray(roi).astype(np.float32).reshape(1, 4)
    feature = im_exfeat(net, im, roi)
    feature = feature[blob_name].squeeze()
    return feature
