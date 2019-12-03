"""
Author: 520Chris
Description: Train a person search network.
"""

import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.optim as optim

from datasets.factory import get_imdb
from models.network import Network
from roi_data_layer.dataloader import DataLoader
from utils.config import cfg, cfg_from_file, get_output_dir


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a person search network')
    parser.add_argument('--gpu', dest='gpu',
                        help='GPU device id to use [0,1,2,3,4,5,6,7,8]',
                        default='0', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--snapshot', dest='previous_state',
                        help='initialize with previous solver state',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    return parser.parse_args()


def init_from_caffe(net):
    dict_new = net.state_dict().copy()
    weight_path = '/home/zjli/Desktop/person_search/pkl1/caffe_model_weights.pkl'
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
    net.labeled_matching_layer.lookup_table = torch.from_numpy(caffe_weights['labeled_matching'][0]).cuda()
    net.unlabeled_matching_layer.queue.data = torch.from_numpy(caffe_weights['unlabeled_matching'][0]).cuda()
    print("Load caffe model successfully!")


def prepare_imdb(name):
    print("Loading image database: %s" % name)
    imdb = get_imdb(name)
    print("Done.")
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('Done.')
    return imdb


if __name__ == '__main__':
    args = parse_args()
    # net = Network()
    # for k, v in net.named_parameters():
    #     print(k)

    # print('Called with args:')
    # print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if not args.randomize:
        # Fix the random seeds (numpy and pytorch) for reproducibility
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)

    imdb = prepare_imdb(args.imdb_name)
    roidb = imdb.roidb
    # print('%s roidb entries' % len(roidb))

    output_dir = get_output_dir(imdb.name)
    print('Output will be saved to `%s`' % output_dir)

    dataloader = DataLoader(roidb)
    net = Network()
    init_from_caffe(net)
    net.cuda()
    optimizer = optim.SGD(net.get_training_params(),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.1)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # for i in range(10):
    #     print(optimizer.state_dict()['param_groups'][0]['lr'])
    #     if False:
    #         optimizer.step()
    #     scheduler.step()

    max_iters = 50000
    iter_size = 2  # accumulated gradient update
    display = 20
    average_loss = 100
    losses = []
    smoothed_loss = 0
    for i in range(max_iters):
        # forward one iteration
        loss = 0
        for _ in range(iter_size):
            blob = dataloader.get_next_minibatch()
            output = net(torch.from_numpy(blob['data']).cuda(),
                         torch.from_numpy(blob['im_info']).cuda(),
                         torch.from_numpy(blob['gt_boxes']).cuda())
            _, _, _, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox, loss_id = output
            loss_iter = (rpn_loss_cls + rpn_loss_bbox + loss_cls + loss_bbox + loss_id) / iter_size
            loss_iter.backward()
            loss += loss_iter

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # adjust learning rate

        if len(losses) < average_loss:
            losses.append(loss)
            size = len(losses)
            smoothed_loss = (smoothed_loss * (size - 1) + loss) / size
        else:
            idx = i % average_loss
            smoothed_loss += (loss - losses[idx]) / average_loss
            losses[idx] = loss

        if i % display == 0:
            print("Iteration [%s / %s]: loss: %.4f" % (i, max_iters, smoothed_loss))
            print("rpn_loss_cls: %.4f, rpn_loss_bbox: %.4f, loss_cls: %.4f, loss_bbox: %.4f, loss_id: %.4f" %
                  (rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox, loss_id))

    torch.save(net, 'net.pth')
