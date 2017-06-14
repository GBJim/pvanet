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
from datasets.vatic import IMDBGroup

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






def train_coco_group(net_params, output_dir, image_set, year,bbox_pred_name="bbox_pred-coco",  CLS_mapper={}, GPU_ID=0, randomize=False, cfg="models/pvanet/lite/train.yml"):
    
    cfg_from_file(cfg)
     
    print('Using config:')
    pprint.pprint(cfg)
    
    #if not randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        #np.random.seed(cfg.RNG_SEED)
        #caffe.set_random_seed(cfg.RNG_SEED)

    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    #imdbs = [VaticData(vatic_name, class_set_name, CLS_mapper=CLS_mapper) for vatic_name in vatic_names]
    #vatic_group = VaticGroup(imdbs)
    
    coco_train = coco("train", year)
    coco_val = coco("val", year)
    
    
    
  
    
    #datasets = [coco_train, coco_val]
    #imdb_group = IMDBGroup(datasets)
    
    
    
    
   
    imdb, roidb = combined_roidb(coco_train)
    
    print '{:d} roidb entries'.format(len(roidb))
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print 'Trained model will be saved to `{:s}`'.format(output_dir)
    
    
    solver, train_pt, caffenet, max_iters, model_name = net_params
    train_net(solver, roidb, output_dir, model_name,
              pretrained_model=caffenet, max_iters=max_iters, bbox_pred_name=bbox_pred_name)
    
    
    
    
    

if __name__ == '__main__':
    
    
    
    output_dir = "models/coco/all"
    
    
    
    '''
    vatic_names = ["YuDa","A1HighwayDay", "B2HighwayNight"]
    mapper = {"van":"car", "truck":"car", "trailer-head":"car",\
              "sedan/suv":"car", "scooter":"motorbike", "bike":"bicycle"}
    class_set_name = "voc"          
    '''
    
    
    
    
    solver = "models/pvanet/lite/coco_solver.prototxt"
    train_pt = "models/pvanet/lite/coco_train.prototxt"
    caffenet = "models/pvanet/lite/test.model"
    max_iters = 20 * 10000
    #max_iters = 20
    output_name = "coco80"
    net_params = (solver, train_pt, caffenet, max_iters, output_name)
    
    
    image_set = "train"
    year="2014"
    GPU_ID = 3
    
   
    train_coco_group(net_params, output_dir,image_set, year, GPU_ID=GPU_ID,  )
    
