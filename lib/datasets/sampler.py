import numpy as np
from torch.utils.data.sampler import Sampler

from utils.config import cfg


class PSSampler(Sampler):
    def __init__(self, dataset):
        Sampler.__init__(self, dataset)
        self.roidb = dataset.roidb

    def __iter__(self):
        if cfg.TRAIN.ASPECT_GROUPING:
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
            perm = inds
        else:
            perm = np.random.permutation(np.arange(len(self.roidb)))
        return iter(perm)

    def __len__(self):
        return len(self.roidb)
