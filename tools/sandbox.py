#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
#from datasets.factory import get_imdb
#from datasets.vatic import VaticData
import datasets.imdb
#from datasets.vatic import VaticGroup
from datasets.coco import coco
from datasets.vatic import VaticData, IMDBGroup
from datasets.pascal_voc_new import pascal_voc
import caffe
import argparse
import pprint
import numpy as np
import sys
import os


def combined_roidb(imdb):
    def get_roidb(imdb):
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    
    roidb = get_roidb(imdb)
  
    return imdb, roidb






def train_coco_group(net_params, output_dir, image_set, year,  CLS_mapper={}, GPU_ID=2, randomize=False, cfg="models/pvanet/lite/train.yml"):
    
    cfg_from_file(cfg)
     
    print('Using config:')
    pprint.pprint(cfg)
    
    #if not randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        #np.random.seed(cfg.RNG_SEED)
        #caffe.set_random_seed(cfg.RNG_SEED)

    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    #
    #vatic_group = VaticGroup(imdbs)
    
    
    
    
    #cocos = [coco("train", "2014"), coco("test", "2014")]
     
    coco_train = coco("train", year)
    coco_val = coco("val", year)
    #global classes
    classes = coco_val._classes
    print(classes)
    
    
    devkit_path = "/root/data/VOCdevkit"
    
    
    mapper = {"tvmonitor":"tv", "sofa":"couch", "aeroplane":"airplane",\
              "motorbike":"motorcycle", "diningtable":"dining table", "pottedplant":"potted plant"}
    
    
    voc2007 = pascal_voc(classes, "trainval", "2007", cls_mapper=mapper, devkit_path=devkit_path)
    voc2012 = pascal_voc(classes, "trainval", "2012", cls_mapper=mapper, devkit_path=devkit_path)
    
    
    
    
    
    vatic_names = ["YuDa","A1HighwayDay", "B2HighwayNight"]
    
    mapper = {"van":"car", "trailer-head":"truck",\
              "sedan/suv":"car", "scooter":"motorcycle", "bike":"bicycle"}
    
    
    
    vatics = [VaticData(vatic_name, "coco", CLS_mapper=mapper) for vatic_name in vatic_names]
    
    datasets = [coco_train, coco_val] + vatics  + [voc2007, voc2012]
    
    #coco_data = coco(image_set, year)
    imdb_group = IMDBGroup(datasets)
    imdb, roidb = combined_roidb(imdb_group)
    
    print '{:d} roidb entries'.format(len(roidb))
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print 'Trained model will be saved to `{:s}`'.format(output_dir)
    
    
    solver, train_pt, caffenet, max_iters, model_name = net_params
    train_net(solver, roidb, output_dir, model_name,
              pretrained_model=caffenet, max_iters=max_iters)
    
    
    
    
    

if __name__ == '__main__':
    
    
    
    output_dir = "models/coco/all"
    
    
    
    
    solver = "models/pvanet/lite/coco_solver.prototxt"
    train_pt = "models/pvanet/lite/coco_train.prototxt"
    caffenet = "models/pvanet/lite/test.model"
    max_iters = 20 * 10000
    output_name = "all-coco80"
    net_params = (solver, train_pt, caffenet, max_iters, output_name)
    
    
    image_set = "train"
    year="2014"
    
    
    train_coco_group(net_params, output_dir,image_set, year)
    
