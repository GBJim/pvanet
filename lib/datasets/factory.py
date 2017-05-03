# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}
from datasets.vatic import VaticData, has_data, data_list
from datasets.vatic import has_data
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
import numpy as np


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not has_data(name):
        print("The available dataset are: {}".format(data_list))
        raise Exception("Dataset: {} does not exist".format(name))
    return VaticData(name)

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
