import argparse
import os.path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.core import tensor2imgs, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from src import *  # noqa
from tools.fuse_conv_bn import fuse_module


def single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    model.eval()
    gallery_det = []
    gallery_feat = []
    probe_feat = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            is_gallery = torch.all(data["proposals"][0] == 0)
            if is_gallery:
                data.pop("proposals")
                det, feat = model(return_loss=False, rescale=not show, **data)
                gallery_det.append(det[0])
                gallery_feat.append(feat)
            else:
                det, feat = model(return_loss=False, rescale=not show, **data)
                probe_feat.append(feat)

        if show or out_dir:
            img_tensor = data["img"][0]
            img_metas = data["img_metas"][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]["img_norm_cfg"])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta["img_shape"]
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta["ori_shape"][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta["filename"])
                else:
                    out_file = None

                model.module.show_result(
                    img_show, det, show=show, out_file=out_file, score_thr=show_score_thr
                )

        batch_size = data["img"][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return gallery_det, gallery_feat, probe_feat


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase the inference speed",
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where painted images will be saved")
    parser.add_argument(
        "--show-score-thr", type=float, default=0.3, help="score threshold (default: 0.3)"
    )
    parser.add_argument("--options", nargs="+", action=DictAction, help="arguments in dict")
    parser.add_argument(
        "--eval_only", action="store_true", help="evaluation with pre-extracted features"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,  # not distributed
        shuffle=False,
    )

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_module(model)
    model.CLASSES = checkpoint["meta"]["CLASSES"]

    if not args.eval_only:
        model = MMDataParallel(model, device_ids=[0])
        gallery_det, gallery_feat, probe_feat = single_gpu_test(
            model, data_loader, args.show, args.show_dir, args.show_score_thr
        )
        mmcv.dump(gallery_det, "gallery_det.pkl")
        mmcv.dump(gallery_feat, "gallery_feat.pkl")
        mmcv.dump(probe_feat, "probe_feat.pkl")
    else:
        gallery_det = mmcv.load("gallery_det.pkl")
        gallery_feat = mmcv.load("gallery_feat.pkl")
        probe_feat = mmcv.load("probe_feat.pkl")

    dataset.evaluate(gallery_det, gallery_feat, probe_feat)


if __name__ == "__main__":
    main()
