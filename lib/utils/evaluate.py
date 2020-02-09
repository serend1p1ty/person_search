import logging
import os.path as osp

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import average_precision_score


def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def evaluate_detections(dataset, gallery_det, threshold=0.5, iou_thresh=0.5, labeled_only=False):
    """
    gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
    threshold (float): filter out gallery detections whose scores below this
    iou_thresh (float): treat as true positive if IoU is above this threshold
    labeled_only (bool): filter out unlabeled background people
    """
    assert dataset.num_images == len(gallery_det)

    roidb = dataset.roidb
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
                ious[i, j] = compute_iou(gt_boxes[i], det[j, :4])
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

    logging.info("{} detection:".format("Labeled only" if labeled_only else "All"))
    logging.info("  Recall = {:.2%}".format(det_rate))
    if not labeled_only:
        logging.info("  AP = {:.2%}".format(ap))


def evaluate_search(
    dataset, gallery_det, gallery_feat, probe_feat, threshold=0.5, gallery_size=100
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
    assert dataset.num_images == len(gallery_det)
    assert dataset.num_images == len(gallery_feat)
    assert len(dataset.probes) == len(probe_feat)

    # TODO: support evaluation on training split
    use_full_set = gallery_size == -1
    fname = "TestG{}".format(gallery_size if not use_full_set else 50)
    protoc = loadmat(osp.join(dataset.root_dir, "annotation/test/train_test", fname + ".mat"))[
        fname
    ].squeeze()

    # mapping from gallery image to (det, feat)
    name_to_det_feat = {}
    for name, det, feat in zip(dataset.image_index, gallery_det, gallery_feat):
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
                    if compute_iou(roi, gt) >= iou_thresh:
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
            for gallery_imname in dataset.image_index:
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

    logging.info("Search ranking:")
    logging.info("  mAP = {:.2%}".format(np.mean(aps)))
    accs = np.mean(accs, axis=0)
    for i, k in enumerate(topk):
        logging.info("  Top-{:2d} = {:.2%}".format(k, accs[i]))
