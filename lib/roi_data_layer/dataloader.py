"""
Author: 520Chris
Description: Data loader for the network.
"""

import numpy as np

from roi_data_layer.minibatch import get_minibatch
from utils.config import cfg


class DataLoader:
    """Shuffle data and get minibatch blobs."""

    def __init__(self, roidb):
        self.roidb = roidb
        self.perm = None  # Index permutation
        self.cur = None  # Current pos
        self.shuffle_roidb_inds()

    def shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            # self.perm = np.arange(len(self.roidb))
            widths = np.array([r["width"] for r in self.roidb])
            heights = np.array([r["height"] for r in self.roidb])
            horz = widths >= heights
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self.perm = inds
        else:
            self.perm = np.random.permutation(np.arange(len(self.roidb)))
        self.cur = 0

    def get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self.cur + cfg.TRAIN.IMS_PER_BATCH >= len(self.roidb):
            self.shuffle_roidb_inds()
        batch_inds = self.perm[self.cur:self.cur + cfg.TRAIN.IMS_PER_BATCH]
        self.cur += cfg.TRAIN.IMS_PER_BATCH
        return batch_inds

    def get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        batch_inds = self.get_next_minibatch_inds()
        minibatch_db = [self.roidb[i] for i in batch_inds]
        blobs = get_minibatch(minibatch_db)
        for key in blobs:
            blobs[key] = blobs[key].astype(np.float32)
        return blobs
