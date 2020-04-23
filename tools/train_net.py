import argparse
import logging
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import _init_paths  # noqa: F401
from datasets.psdb import PSDB
from datasets.sampler import PSSampler
from models.network import Network
from utils.config import cfg, cfg_from_file
from utils.utils import init_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument(
        "--gpu", default=-1, type=int, help="GPU device id to use. Default: -1, means using CPU"
    )
    parser.add_argument(
        "--epoch", default=5, type=int, help="Number of epochs to train. Default: 5"
    )
    parser.add_argument(
        "--weights",
        help="Initialize with pretrained model weights. "
        + "Default: $ROOT/data/pretrained_model/resnet50_caffe.pth",
    )
    parser.add_argument(
        "--checkpoint", help="Initialize with previous solver state. Default: None",
    )
    parser.add_argument("--cfg", help="Optional config file. Default: None")
    parser.add_argument(
        "--data_dir", help="The directory that saving experimental data. Default: $ROOT/data",
    )
    parser.add_argument(
        "--dataset", default="psdb_train", help="Dataset to train on. Default: psdb_train"
    )
    parser.add_argument(
        "--rand", action="store_true", help="Do not use a fixed seed. Default: False"
    )
    parser.add_argument("--solver", default="sgd", help="Training optimizer. Default: sgd")
    parser.add_argument("--tbX", action="store_true", help="Enable tensorboardX. Default: False")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.cfg:
        cfg_from_file(args.cfg)
    if args.data_dir:
        cfg.DATA_DIR = osp.abspath(args.data_dir)
    if args.weights is None and args.checkpoint is None:
        args.weights = osp.join(cfg.DATA_DIR, "pretrained_model", "resnet50_caffe.pth")

    init_logger("train.log")
    logging.info("Called with args:\n" + str(args))

    if not args.rand:
        # Fix the random seeds (numpy and pytorch) for reproducibility
        logging.info("Set to none random mode.")
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.RNG_SEED)

    output_dir = osp.join(cfg.DATA_DIR, "trained_model")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert args.dataset in ["psdb_train", "psdb_test"], "Unknown dataset: %s" % args.dataset
    dataset = PSDB(args.dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=PSSampler(dataset))
    logging.info("Loaded dataset: %s" % args.dataset)

    # Initialize model
    net = Network()
    if args.weights:
        state_dict = torch.load(args.weights)
        net.load_state_dict({k: v for k, v in state_dict.items() if k in net.state_dict()})
        logging.info("Loaded pretrained model from: %s" % args.weights)

    # Initialize optimizer
    lr = cfg.TRAIN.LEARNING_RATE
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    params = []
    for k, v in net.named_parameters():
        if v.requires_grad:
            if "BN" in k:
                params += [{"params": [v], "lr": lr, "weight_decay": 0}]
            elif "bias" in k:
                params += [{"params": [v], "lr": 2 * lr, "weight_decay": 0}]
            else:
                params += [{"params": [v], "lr": lr, "weight_decay": weight_decay}]
    if args.solver == "sgd":
        optimizer = optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    elif args.solver == "adam":
        optimizer = optim.Adam(params)
    else:
        raise KeyError("Only support sgd and adam.")

    # Training settings
    start_epoch = 0
    display = 20  # Display the loss every `display` steps
    lr_decay_by_epoch = False  # True: decay by epoch, otherwise by step
    lr_decay_epoch = 4  # Decay the learning rate every `lr_decay_epoch` epochs
    lr_decay_step = 40000  # Decay the learning rate every `lr_decay_step` steps
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)
    iter_size = 2  # Each update use accumulated gradient by `iter_size` iterations
    use_caffe_smooth_loss = True
    average_loss = 100  # Be used to calculate caffe smoothed loss

    # Load checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        logging.info("Loaded checkpoint from: %s" % args.checkpoint)

    device = torch.device("cuda:%s" % args.gpu if args.gpu != -1 else "cpu")
    net.to(device)

    # Use tensorboardX to visualize experimental results
    if args.tbX:
        from tensorboardX import SummaryWriter

        tb_log_path = osp.join(cfg.DATA_DIR, "tb_logs")
        logger = SummaryWriter(tb_log_path)

    net.train()
    start = time.time()
    accumulated_step = 0
    loss = 0
    losses = []
    ave_loss = 0
    smoothed_loss = 0
    real_steps_per_epoch = int(len(dataloader) / iter_size)
    for epoch in range(start_epoch, args.epoch):
        # Do learning rate decay
        if lr_decay_by_epoch:
            if epoch % lr_decay_epoch == 0 and epoch:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 0.1 * param_group["lr"]

        for step, data in enumerate(dataloader):
            real_step = int(step / iter_size)
            img = data[0].to(device)
            img_info = data[1][0].to(device)
            gt_boxes = data[2][0].to(device)

            total_steps = epoch * real_steps_per_epoch + real_step
            if total_steps == 50000:
                save_name = os.path.join(output_dir, "checkpoint_step_50000.pth")
                save_dict = {
                    "step": 50000,
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if not lr_decay_by_epoch:
                    save_dict["scheduler"] = scheduler.state_dict()
                torch.save(save_dict, save_name)

            _, _, _, _, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox, loss_oim = net(
                img, img_info, gt_boxes
            )
            loss_iter = (rpn_loss_cls + rpn_loss_bbox + loss_cls + loss_bbox + loss_oim) / iter_size
            loss += loss_iter.item()
            loss_iter.backward()

            accumulated_step += 1
            if accumulated_step == iter_size:
                optimizer.step()
                optimizer.zero_grad()

                # Adjust learning rate every real step
                if not lr_decay_by_epoch:
                    scheduler.step()

                if use_caffe_smooth_loss:
                    if len(losses) < average_loss:
                        losses.append(loss)
                        size = len(losses)
                        smoothed_loss = (smoothed_loss * (size - 1) + loss) / size
                    else:
                        idx = real_step % average_loss
                        smoothed_loss += (loss - losses[idx]) / average_loss
                        losses[idx] = loss
                else:
                    ave_loss += loss

                loss = 0
                accumulated_step = 0

                if real_step % display == 0:
                    if use_caffe_smooth_loss:
                        display_loss = smoothed_loss
                    else:
                        display_loss = ave_loss / display if real_step > 0 else ave_loss
                        ave_loss = 0

                    logging.info(
                        (
                            "\n--------------------------------------------------------------\n"
                            + "epoch: [%s / %s], iteration: [%s / %s], loss: %.4f\n"
                            + "time cost: %.2f seconds, learning rate: %s\n"
                            + "--------------------------------------------------------------"
                        )
                        % (
                            epoch,
                            args.epoch - 1,
                            real_step,
                            real_steps_per_epoch - 1,
                            display_loss,
                            time.time() - start,
                            optimizer.param_groups[0]["lr"],
                        )
                    )

                    start = time.time()

                    if args.tbX:
                        log_info = {
                            "loss": display_loss,
                            "rpn_loss_cls": rpn_loss_cls,
                            "rpn_loss_bbox": rpn_loss_bbox,
                            "loss_cls": loss_cls,
                            "loss_bbox": loss_bbox,
                            "loss_oim": loss_oim,
                        }
                        logger.add_scalars("Train/Loss", log_info, total_steps)

        # Save checkpoint every epoch
        save_name = os.path.join(output_dir, "checkpoint_epoch_%s.pth" % epoch)
        save_dict = {"epoch": epoch, "model": net.state_dict(), "optimizer": optimizer.state_dict()}
        if not lr_decay_by_epoch:
            save_dict["scheduler"] = scheduler.state_dict()
        torch.save(save_dict, save_name)

    if args.tbX:
        logger.close()
