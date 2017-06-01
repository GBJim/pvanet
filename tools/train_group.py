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
from datasets.vatic import VaticData
import datasets.imdb
from datasets.vatic import VaticGroup
import caffe
import argparse
import pprint
import numpy as np
import sys
import os

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--output', dest='output',
                        help='model name of output caffemodel',
                        default=None, type=str)
    
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def combined_roidb(imdb):
    def get_roidb(imdb):
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    
    roidb = get_roidb(imdb)
  
    return imdb, roidb






def train_group(net_params, vatic_names, class_set_name , output_dir,  CLS_mapper={}, GPU_ID=0, \
                randomize=False, cfg="models/pvanet/cfgs/train.yml"):
    
    cfg_from_file(cfg)
     
    print('Using config:')
    pprint.pprint(cfg)
    
    #if not randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        #np.random.seed(cfg.RNG_SEED)
        #caffe.set_random_seed(cfg.RNG_SEED)

    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    imdbs = [VaticData(vatic_name, class_set_name, CLS_mapper=CLS_mapper) for vatic_name in vatic_names]
    vatic_group = VaticGroup(imdbs)

    imdb, roidb = combined_roidb(vatic_group)
    print '{:d} roidb entries'.format(len(roidb))
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print 'Trained model will be saved to `{:s}`'.format(output_dir)
    
    
    solver, train_pt, caffenet, max_iters, model_name = net_params
    train_net(solver, roidb, output_dir, model_name,
              pretrained_model=caffenet, max_iters=max_iters)
    
    
    
    
    

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    IMDB_NAMES = ["chruch_street", "YuDa"]
    imdbs = [(imdb_name) for imdb_name in IMDB_NAMES]

    vatic_group = VaticGroup(imdbs)
        
        
    imdb, roidb = combined_roidb(vatic_group)
    print '{:d} roidb entries'.format(len(roidb))

    #output_dir = get_output_dir(imdb)
    output_dir = "./models/trained"
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, roidb, output_dir, args.output,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
