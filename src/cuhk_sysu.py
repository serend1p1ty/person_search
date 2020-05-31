import os
import os.path as osp
import pickle as pk

import numpy as np
from PIL import Image
from scipy.io import loadmat
from sklearn.metrics import average_precision_score

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


def pickle(data, file_path):
    with open(file_path, "wb") as f:
        pk.dump(data, f, pk.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, "rb") as f:
        data = pk.load(f)
    return data


class CUHKSYSULoader:
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_path = osp.join(self.data_root, "Image", "SSM")
        self.image_indexes = self.load_image_indexes()
        self.roidb = self.load_roidb()
        if split == "test":
            self.probes = self.load_probes()

    @property
    def num_images(self):
        return len(self.image_indexes)

    def image_path_at(self, i):
        image_path = osp.join(self.data_path, self.image_indexes[i])
        assert osp.isfile(image_path), "Path does not exist: %s" % image_path
        return image_path

    def load_image_indexes(self):
        """Load the image indexes for training / test."""
        # Test images
        test = loadmat(osp.join(self.data_root, "annotation", "pool.mat"))
        test = test["pool"].squeeze()
        test = [str(a[0]) for a in test]
        if self.split == "test":
            return test

        # All images
        all_imgs = loadmat(osp.join(self.data_root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]

        # Training images = all images - test images
        train = list(set(all_imgs) - set(test))
        train.sort()
        return train

    def load_probes(self):
        """Load the list of (img, roi) for probes."""
        protocol = loadmat(osp.join(self.data_root, "annotation/test/train_test/TestG50.mat"))
        protocol = protocol["TestG50"].squeeze()
        probes = []
        for item in protocol["Query"]:
            im_name = osp.join(self.data_path, str(item["imname"][0, 0][0]))
            roi = item["idlocate"][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]
            size = Image.open(im_name).size
            probes.append((im_name, roi, size[1], size[0]))
        return probes

    def load_roidb(self):
        """Load the ground-truth roidb for each image.

        The roidb of each image is a dictionary that has the following keys:
            gt_boxes (ndarray[N, 4]): all ground-truth boxes in (x1, y1, x2, y2) format
            gt_pids (ndarray[N]): person IDs of these ground-truth boxes
            img_path (str)
            width (int)
            height (int)
        """
        cache_path = osp.join(self.data_root, "cache")
        if not osp.exists(cache_path):
            os.makedirs(cache_path)
        cache_file = osp.join(cache_path, self.split + "_roidb.pkl")
        if osp.isfile(cache_file):
            return unpickle(cache_file)

        # Load all images and build a dict from image to boxes
        all_imgs = loadmat(osp.join(self.data_root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        for im_name, _, boxes in all_imgs:
            im_name = str(im_name[0])
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, "Warning: %s has no valid boxes." % im_name
            boxes = boxes[valid_index]
            name_to_boxes[im_name] = boxes.astype(np.int32)
            name_to_pids[im_name] = -1 * np.ones(boxes.shape[0], dtype=np.int32)

        def set_box_pid(boxes, box, pids, pid):
            for i in range(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return
            print("Person: %s, box: %s cannot find in images." % (pid, box))

        # Load all the train / test persons and label their pids from 0 to N - 1
        # Assign pid = -1 for unlabeled background people
        if self.split == "train":
            train = loadmat(osp.join(self.data_root, "annotation/test/train_test/Train.mat"))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):
                scenes = item[0, 0][2].squeeze()
                for im_name, box, _ in scenes:
                    im_name = str(im_name[0])
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index)
        else:
            test = loadmat(osp.join(self.data_root, "annotation/test/train_test/TestG50.mat"))
            test = test["TestG50"].squeeze()
            for index, item in enumerate(test):
                # query
                im_name = str(item["Query"][0, 0][0][0])
                box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index)

                # gallery
                gallery = item["Gallery"].squeeze()
                for im_name, box, _ in gallery:
                    im_name = str(im_name[0])
                    if box.size == 0:
                        break
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index)

        # Construct the roidb
        roidb = []
        for i, im_name in enumerate(self.image_indexes):
            boxes = name_to_boxes[im_name]
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            pids = name_to_pids[im_name]
            size = Image.open(self.image_path_at(i)).size
            roidb.append(
                {
                    "gt_boxes": boxes,
                    "gt_pids": pids,
                    "img_path": self.image_path_at(i),
                    "height": size[1],
                    "width": size[0],
                }
            )
        pickle(roidb, cache_file)
        print("Save ground-truth roidb to: %s" % cache_file)
        return roidb


@DATASETS.register_module
class CUHKSYSU(CustomDataset):
    CLASSES = ("person",)

    def __init__(self, **kwargs):
        data_root = kwargs.pop("data_root")
        self.split = kwargs.pop("split")
        self.loader = CUHKSYSULoader(data_root, self.split)
        super(CUHKSYSU, self).__init__(**kwargs)
        print("Loaded cuhk_sysu dataset (split=%s) from %s" % (self.split, data_root))

    def load_annotations(self, ann_file):
        data_infos = []
        for i, image_index in enumerate(self.loader.image_indexes):
            roidb = self.loader.roidb[i]
            new_entry = {
                "id": image_index,
                "filename": roidb["img_path"],
                "width": roidb["width"],
                "height": roidb["height"],
            }
            if self.split == "train":
                new_entry["ann"] = {
                    "bboxes": roidb["gt_boxes"].astype(np.float32),
                    "labels": np.array([0] * roidb["gt_boxes"].shape[0]),
                    "person_ids": roidb["gt_pids"].astype(np.int64),
                }
            data_infos.append(new_entry)

        if self.split == "test":
            for filename, _, height, width in self.loader.probes:
                data_infos.append(
                    {
                        "id": osp.basename(filename),
                        "filename": filename,
                        "width": width,
                        "height": height,
                    }
                )
        return data_infos

    def load_proposals(self, proposal_file):
        proposals = []
        for _ in self.loader.image_indexes:
            proposals.append(np.array([[0, 0, 0, 0]], dtype=np.float32))
        for _, roi, _, _ in self.loader.probes:
            proposals.append(roi[None].astype(np.float32))
        return proposals

    def evaluate(self, gallery_det, gallery_feat, probe_feat):
        self.evaluate_detection(gallery_det)
        self.evaluate_detection(gallery_det, labeled_only=True)
        self.evaluate_search(gallery_det, gallery_feat, probe_feat)

    @staticmethod
    def compute_iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter * 1.0 / union

    def evaluate_detection(self, gallery_det, threshold=0.5, iou_thresh=0.5, labeled_only=False):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        threshold (float): filter out gallery detections whose scores below this
        iou_thresh (float): treat as true positive if IoU is above this threshold
        labeled_only (bool): filter out unlabeled background people
        """
        assert self.loader.num_images == len(gallery_det)

        roidb = self.loader.roidb
        y_true, y_score = [], []
        count_gt, count_tp = 0, 0
        for gt, det in zip(roidb, gallery_det):
            gt_boxes = gt["gt_boxes"]
            if labeled_only:
                inds = np.where(gt["gt_pids"].ravel() > 0)[0]
                if len(inds) == 0:
                    continue
                gt_boxes = gt_boxes[inds]
            det = np.asarray(det)
            inds = np.where(det[:, 4].ravel() >= threshold)[0]
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in range(num_gt):
                for j in range(num_det):
                    ious[i, j] = self.compute_iou(gt_boxes[i], det[j, :4])
            tfmat = ious >= iou_thresh
            # for each det, keep only the largest iou of all the gt
            for j in range(num_det):
                largest_ind = np.argmax(ious[:, j])
                for i in range(num_gt):
                    if i != largest_ind:
                        tfmat[i, j] = False
            # for each gt, keep only the largest iou of all the det
            for i in range(num_gt):
                largest_ind = np.argmax(ious[i, :])
                for j in range(num_det):
                    if j != largest_ind:
                        tfmat[i, j] = False
            for j in range(num_det):
                y_score.append(det[j, -1])
                y_true.append(tfmat[:, j].any())
            count_tp += tfmat.sum()
            count_gt += num_gt

        det_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * det_rate

        print("{} detection:".format("Labeled only" if labeled_only else "All"))
        print("  Recall = {:.2%}".format(det_rate))
        if not labeled_only:
            print("  AP = {:.2%}".format(ap))

    def evaluate_search(
        self, gallery_det, gallery_feat, probe_feat, threshold=0.5, gallery_size=100
    ):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        gallery_feat (list of ndarray): n_det x D features per image
        probe_feat (list of ndarray): D dimensional features per probe image
        threshold (float): filter out gallery detections whose scores below this
        gallery_size (int): gallery size [-1, 50, 100, 500, 1000, 2000, 4000]
                            -1 for using full set
        dump_json (str): Path to save the results as a JSON file or None
        """
        dataset = self.loader
        assert dataset.num_images == len(gallery_det)
        assert dataset.num_images == len(gallery_feat)
        assert len(dataset.probes) == len(probe_feat)

        gallery_feat = [i.cpu().numpy() for i in gallery_feat]
        probe_feat = [i.cpu().numpy() for i in probe_feat]

        # TODO: support evaluation on training split
        use_full_set = gallery_size == -1
        fname = "TestG{}".format(gallery_size if not use_full_set else 50)
        protoc = loadmat(osp.join(dataset.data_root, "annotation/test/train_test", fname + ".mat"))[
            fname
        ].squeeze()

        # mapping from gallery image to (det, feat)
        name_to_det_feat = {}
        for name, det, feat in zip(dataset.image_indexes, gallery_det, gallery_feat):
            scores = det[:, 4].ravel()
            inds = np.where(scores >= threshold)[0]
            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds])

        aps = []
        accs = []
        topk = [1, 5, 10]
        ret = {"image_root": dataset.data_path, "results": []}
        for i in range(len(dataset.probes)):
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0
            # Get L2-normalized feature vector
            feat_p = probe_feat[i].ravel()
            # Ignore the probe image
            probe_imname = str(protoc["Query"][i]["imname"][0, 0][0])
            probe_roi = protoc["Query"][i]["idlocate"][0, 0][0].astype(np.int32)
            probe_roi[2:] += probe_roi[:2]
            probe_gt = []
            tested = set([probe_imname])
            # 1. Go through the gallery samples defined by the protocol
            for item in protoc["Gallery"][i].squeeze():
                gallery_imname = str(item[0][0])
                # some contain the probe (gt not empty), some not
                gt = item[1][0].astype(np.int32)
                count_gt += gt.size > 0
                # compute distance between probe and gallery dets
                if gallery_imname not in name_to_det_feat:
                    continue
                det, feat_g = name_to_det_feat[gallery_imname]
                # get L2-normalized feature matrix NxD
                assert feat_g.size == np.prod(feat_g.shape[:2])
                feat_g = feat_g.reshape(feat_g.shape[:2])
                # compute cosine similarities
                sim = feat_g.dot(feat_p).ravel()
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                if gt.size > 0:
                    w, h = gt[2], gt[3]
                    gt[2:] += gt[:2]
                    probe_gt.append({"img": str(gallery_imname), "roi": map(float, list(gt))})
                    iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]
                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if self.compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                y_true.extend(list(label))
                y_score.extend(list(sim))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))
                tested.add(gallery_imname)
            # 2. Go through the remaining gallery images if using full set
            if use_full_set:
                for gallery_imname in dataset.image_indexes:
                    if gallery_imname in tested:
                        continue
                    if gallery_imname not in name_to_det_feat:
                        continue
                    det, feat_g = name_to_det_feat[gallery_imname]
                    # get L2-normalized feature matrix NxD
                    assert feat_g.size == np.prod(feat_g.shape[:2])
                    feat_g = feat_g.reshape(feat_g.shape[:2])
                    # compute cosine similarities
                    sim = feat_g.dot(feat_p).ravel()
                    # guaranteed no target probe in these gallery images
                    label = np.zeros(len(sim), dtype=np.int32)
                    y_true.extend(list(label))
                    y_score.extend(list(sim))
                    imgs.extend([gallery_imname] * len(sim))
                    rois.extend(list(det))
            # 3. Compute AP for this probe (need to scale by recall rate)
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            assert count_tp <= count_gt
            recall_rate = count_tp * 1.0 / count_gt
            ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
            aps.append(ap)
            inds = np.argsort(y_score)[::-1]
            y_score = y_score[inds]
            y_true = y_true[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])
            # 4. Save result for JSON dump
            new_entry = {
                "probe_img": str(probe_imname),
                "probe_roi": map(float, list(probe_roi)),
                "probe_gt": probe_gt,
                "gallery": [],
            }
            # only save top-10 predictions
            for k in range(10):
                new_entry["gallery"].append(
                    {
                        "img": str(imgs[inds[k]]),
                        "roi": map(float, list(rois[inds[k]])),
                        "score": float(y_score[k]),
                        "correct": int(y_true[k]),
                    }
                )
            ret["results"].append(new_entry)

        print("Search ranking:")
        print("  mAP = {:.2%}".format(np.mean(aps)))
        accs = np.mean(accs, axis=0)
        for i, k in enumerate(topk):
            print("  Top-{:2d} = {:.2%}".format(k, accs[i]))
