"""
Author: Ross Girshick
Description: Factory method for easily getting imdbs by name.
"""

from datasets.psdb import PSDB

__sets = {}

# Set up person search database
for split in ["train", "test"]:
    name = "psdb_%s" % split
    __sets[name] = lambda name=name: PSDB(name)


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError("Unknown dataset: %s" % name)
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
