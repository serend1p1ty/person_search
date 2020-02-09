import argparse
import logging
import os.path as osp

import cv2
import torch
from tqdm import tqdm

import _init_paths  # noqa: F401
from datasets.psdb import PSDB
from models.network import Network
from utils.config import cfg, cfg_from_file
from utils.evaluate import evaluate_detections, evaluate_search
from utils.utils import init_logger, pickle, unpickle


def parse_args():
    parser = argparse.ArgumentParser(description="Test the person search network.")
    parser.add_argument(
        "--gpu", default=-1, type=int, help="GPU device id to use. Default: -1, means using CPU"
    )
    parser.add_argument("--checkpoint", help="The checkpoint to be tested. Default: None")
    parser.add_argument("--cfg", help="Optional config file. Default: None")
    parser.add_argument(
        "--data_dir", help="The directory that saving experimental data. Default: $ROOT/data",
    )
    parser.add_argument(
        "--dataset", default="psdb_test", help="Dataset to test on. Default: psdb_test"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Evaluation with pre extracted features. Default: False",
    )
    return parser.parse_args()


def detect_and_exfeat(net, dataset, threshold=0.05):
    """
    Detect and extract features for each image in dataset.
    """
    num_images = dataset.num_images
    all_boxes = []
    all_features = []
    for i in tqdm(range(num_images)):
        img = cv2.imread(dataset.image_path_at(i))
        detection, feat = net.inference(img, threshold=threshold)
        all_boxes.append(detection.cpu().numpy())
        all_features.append(feat.cpu().numpy())
    return all_boxes, all_features


def exfeat(net, probes):
    """
    Extract the features of given probe RoI.
    """
    num_images = len(probes)
    all_features = []
    for i in tqdm(range(num_images)):
        im_name, roi = probes[i]
        img = cv2.imread(im_name)
        feat = net.inference(img, roi)
        all_features.append(feat.cpu().numpy())
    return all_features


if __name__ == "__main__":
    args = parse_args()

    if args.cfg:
        cfg_from_file(args.cfg)
    if args.checkpoint is None:
        raise KeyError("--checkpoint option can not be empty.")
    if args.data_dir:
        cfg.DATA_DIR = osp.abspath(args.data_dir)

    init_logger("test.log")
    logging.info("Called with args:\n" + str(args))

    dataset = PSDB(args.dataset)
    logging.info("Loaded dataset: %s" % args.dataset)

    net = Network()
    checkpoint = torch.load(osp.abspath(args.checkpoint))
    net.load_state_dict(checkpoint["model"])
    logging.info("Loaded checkpoint from: %s" % args.checkpoint)
    net.eval()
    device = torch.device("cuda:%s" % args.gpu if args.gpu != -1 else "cpu")
    net.to(device)

    save_path = osp.join(cfg.DATA_DIR, "cache")
    if args.eval_only:
        gboxes = unpickle(osp.join(save_path, "gallery_detections.pkl"))
        gfeatures = unpickle(osp.join(save_path, "gallery_features.pkl"))
        pfeatures = unpickle(osp.join(save_path, "probe_features.pkl"))
    else:
        # 1. Detect and extract features from all the gallery images in the dataset
        gboxes, gfeatures = detect_and_exfeat(net, dataset)

        # 2. Only extract features of given probe RoI
        pfeatures = exfeat(net, dataset.probes)

        pickle(gboxes, osp.join(save_path, "gallery_detections.pkl"))
        pickle(gfeatures, osp.join(save_path, "gallery_features.pkl"))
        pickle(pfeatures, osp.join(save_path, "probe_features.pkl"))

    # Evaluate
    evaluate_detections(dataset, gboxes, threshold=0.5)
    evaluate_detections(dataset, gboxes, threshold=0.5, labeled_only=True)
    evaluate_search(dataset, gboxes, gfeatures, pfeatures, threshold=0.5, gallery_size=100)
