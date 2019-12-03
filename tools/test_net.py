"""
Author: https://github.com/ShuangLI59/person_search.git
Description: Evaluate network in psdb_test dataset.
"""

import pickle

import torch

from datasets.factory import get_imdb
from models.network import Network
from test_gallery import detect_and_exfeat
from test_probe import exfeat
from utils.config import cfg
from utils import unpickle


def init_from_caffe(net):
    dict_new = net.state_dict().copy()
    weight_path = '/home/zjli/Desktop/person_search/pkl/caffe_model_weights.pkl'
    caffe_weights = pickle.load(open(weight_path, "rb"), encoding='latin1')
    for k in net.state_dict():
        splits = k.split('.')

        # Layer name mapping
        if splits[-2] == 'rpn_conv':
            name = 'rpn_conv/3x3'
        elif splits[-2] == 'cls_score':
            name = 'det_score'
        elif splits[-2] in ['rpn_cls_score', 'rpn_bbox_pred', 'bbox_pred', 'feat_lowdim']:
            name = splits[-2]
        else:
            name = 'caffe.' + splits[-2]

        if name not in caffe_weights:
            print("Layer: %s not found" % k)
            continue

        if splits[-1] == 'weight':  # For BN, weight is scale
            dict_new[k] = torch.from_numpy(caffe_weights[name][0]).reshape(dict_new[k].shape)
        elif splits[-1] == 'bias':  # For BN, bias is shift
            dict_new[k] = torch.from_numpy(caffe_weights[name][1]).reshape(dict_new[k].shape)
        elif splits[-1] == 'running_mean':
            dict_new[k] = torch.from_numpy(caffe_weights[name][2]).reshape(dict_new[k].shape)
        elif splits[-1] == 'running_var':
            dict_new[k] = torch.from_numpy(caffe_weights[name][3]).reshape(dict_new[k].shape)
        elif splits[-1] == 'num_batches_tracked':  # num_batches_tracked is unuseful in test phase
            continue
        else:
            print("Layer: %s not found" % k)
            continue

    net.load_state_dict(dict_new)
    print("Load caffe model successfully!")


def main():
    cfg.TEST.NMS = 0.4
    cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    imdb = get_imdb('psdb_test')

    # 1. Detect and extract features from all the gallery images in the imdb
    net = torch.load('net.pth')
    # net = Network()
    # init_from_caffe(net)
    net.eval()
    net.cuda()
    gboxes, gfeatures = detect_and_exfeat(net, imdb)

    # 2. Only extract features from given probe rois
    pfeatures = exfeat(net, imdb.probes)

    # Save
    # from utils import pickle
    # pickle(gboxes, 'gallery_detections.pkl')
    # pickle(gfeatures, 'gallery_features.pkl')
    # pickle(pfeatures, 'probe_features.pkl')

    # gboxes = pickle.load(open('./pkl/gallery_detections.pkl', "rb"), encoding='latin1')
    # gfeatures = pickle.load(open('./pkl/gallery_features.pkl', "rb"), encoding='latin1')
    # pfeatures = pickle.load(open('./pkl/probe_features.pkl', "rb"), encoding='latin1')

    # gboxes = unpickle('gallery_detections.pkl')
    # gfeatures = unpickle('gallery_features.pkl')
    # pfeatures = unpickle('probe_features.pkl')

    # Evaluate
    imdb.evaluate_detections(gboxes, det_thresh=0.5)
    imdb.evaluate_detections(gboxes, det_thresh=0.5, labeled_only=True)
    imdb.evaluate_search(gboxes, gfeatures['feat'], pfeatures['feat'], det_thresh=0.5,
                         gallery_size=100)


if __name__ == "__main__":
    main()
