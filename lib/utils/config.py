import os.path as osp

import numpy as np
import yaml
from easydict import EasyDict as edict

cfg = edict()


######################
#  Training options  #
######################

cfg.TRAIN = edict()

# Learning rate
cfg.TRAIN.LEARNING_RATE = 0.001
# Weight decay
cfg.TRAIN.WEIGHT_DECAY = 0.0005
# Momentum
cfg.TRAIN.MOMENTUM = 0.9
# Iterations between snapshots
cfg.TRAIN.SNAPSHOT_ITERS = 10000
# Images to use per minibatch
cfg.TRAIN.IMS_PER_BATCH = 1
# Minibatch size (number of regions of interest [RoIs])
cfg.TRAIN.BATCH_SIZE = 128
# Fraction of minibatch that is labeled foreground (i.e. class > 0)
cfg.TRAIN.FG_FRACTION = 0.25
# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
cfg.TRAIN.FG_THRESH = 0.5
# Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))
cfg.TRAIN.BG_THRESH_HI = 0.5
cfg.TRAIN.BG_THRESH_LO = 0.0
# Use horizontally-flipped images during training
cfg.TRAIN.USE_FLIPPED = True
# Deprecated (inside weights)
cfg.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
# cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
cfg.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
cfg.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
# Make minibatches from images that have similar aspect ratios (i.e. both tall and
# thin or both short and wide) in order to avoid wasting computation on zero-padding.
cfg.TRAIN.ASPECT_GROUPING = True
# IOU >= thresh: positive example
cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
cfg.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# Max number of foreground examples
cfg.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
cfg.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
cfg.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
cfg.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
cfg.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
cfg.TRAIN.RPN_MIN_SIZE = 16
# Deprecated (outside weights)
cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


#####################
#  Testing options  #
#####################

cfg.TEST = edict()

# Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)
cfg.TEST.NMS = 0.4
# NMS threshold used on RPN proposals
cfg.TEST.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
cfg.TEST.RPN_PRE_NMS_TOP_N = 6000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
cfg.TEST.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
cfg.TEST.RPN_MIN_SIZE = 16


##########
#  MISC  #
##########

# The scale to be used during training, which is the pixel size of an image's shortest side
cfg.SCALE = 600
# Max pixel size of the longest side of a scaled input image
cfg.MAX_SIZE = 1000
# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
cfg.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
# For reproducibility
cfg.RNG_SEED = 3
# Root directory of project
cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
# Data directory
cfg.DATA_DIR = osp.join(cfg.ROOT_DIR, "data")
# Default GPU device id
cfg.GPU_ID = 0
# Size of the pooled region after RoI pooling
cfg.POOLING_SIZE = 14
# Anchor scales for RPN
cfg.ANCHOR_SCALES = [8, 16, 32]
# Anchor ratios for RPN
cfg.ANCHOR_RATIOS = [0.5, 1, 2]
# Feature stride for RPN
cfg.FEAT_STRIDE = [16]


def merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering
    the options in b whenever they are also specified in a.
    """
    if not isinstance(a, edict):
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("%s is not a valid config key" % k)

        # The types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(
                    "Type mismatch (%s vs. %s) for config key: %s" % (type(b[k]), type(v), k)
                )

        # Recursively merge dicts
        if isinstance(v, edict):
            merge_a_into_b(a[k], b[k])
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, "r") as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    merge_a_into_b(yaml_cfg, cfg)
