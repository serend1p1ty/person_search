"""
Author: Ross Girshick
Last editor: 520Chris
Description: Image database used by network.
"""

import os
import os.path as osp

from PIL import Image

from utils.config import cfg


class IMDB:
    """A common image database. Inherit this class to develop a new one."""

    def __init__(self, name):
        self.name = name
        self.classes = []
        self.image_index = []
        self.probes = []
        self.roidb = []

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_images(self):
        return len(self.image_index)

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, "cache"))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def image_path_at(self, i):
        raise NotImplementedError

    def append_flipped_images(self):
        num_images = self.num_images
        widths = [Image.open(self.image_path_at(i)).size[0] for i in range(self.num_images)]
        for i in range(num_images):
            boxes = self.roidb[i]["boxes"].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {
                "boxes": boxes,
                "gt_pids": self.roidb[i]["gt_pids"],
                "image": self.roidb[i]["image"],
                "height": self.roidb[i]["height"],
                "width": self.roidb[i]["width"],
                "flipped": True,
            }
            self.roidb.append(entry)
        self.image_index = self.image_index * 2
