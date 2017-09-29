"""
COCO 2014 train + coco 2014 val + Vatic[Yuda, A1Highwayday, A2HighwayNight]
"""



"""This is an example of fine tuning PVA-NET through IMDB moudle that combines different dataset together
Most of the codes are modified from Faster R-CNN

"""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import datasets.imdb
from datasets.coco import coco
from datasets.vatic import VaticData, IMDBGroup
from datasets.pascal_voc_new import pascal_voc
from datasets.config import CLASS_SETS
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



def prepare_data():
     #Set the training configuration first
    cfg_path="models/pvanet/lite/train_1K.yml"
    cfg_from_file(cfg_path)
    
    """
     1. PREPARING DATASET
    """


    #Firstly, prepare the dataset for fine-tuning 
    #Different kind of dataset is wrapped by the IMDB class, originally designed by Ross Girshick    
    
    #You need to put coco data directory(soft-link works as well) under the PVA-NET directory
    #COCO IMDB needs two parameter: data-split and year    
    #coco_train = coco("train", "20SNAPSHOT_ITERS: 6014")   
    main_classes = CLASS_SETS["NCTU-vehicles"]
  
    mapper = { "trailer-head":"truck", 'person': '__background__', 'scooter': 'motorcycle', 'bike':'bicycle'}  
    vatic_names = ["A1HighwayDay", 'B2HighwayNight', "pickup", "tanktruck", "van", "PU_Van"]
    vatics = [VaticData(vatic_name, main_classes, CLS_mapper=mapper) for vatic_name in vatic_names]        
    imdb_group = IMDBGroup(vatics)       
    imdb, roidb = combined_roidb(imdb_group)
    total_len = float(len(imdb_group.gt_roidb()))
    print(total_len)
   
    return roidb


def finetune(net_params, roidb, GPU_ID=1):

   
    solver, train_pt, caffenet,bbox_pred_name, max_iters, output_dir, output_prefix = net_params
    
        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print 'Trained model will be saved to `{:s}`'.format(output_dir)
    
    
    
    
    
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)  
    
    
    train_net(solver, roidb, output_dir, output_prefix,
              pretrained_model=caffenet, max_iters=max_iters, bbox_pred_name="bbox_pred-coco")
    
    
    


if __name__ == '__main__':
    
    
    #Prepare roidb
    roidb = prepare_data()
    
   
    
       
    # Set each training parameter    
    solver = "/root/pva-faster-rcnn/models/pvanet/lite/seven_solver.prototxt"
    train_pt = "/root/pva-faster-rcnn/models/pvanet/lite/seven_train.prototxt"
    caffenet = "/root/pva-faster-rcnn/models/pvanet/lite/test.model"
    
    #The bbox_pred_name is used to specify the new name of bbox_pred layer in the modified prototxt. bbox_pred layer is handeled differentlly in the snapshooting procedure for the purpose of bbox normalization. In order to prevent sanpshotting the un-tuned bbox_pred layer, we need to specify the new name.  
    bbox_pred_name = "bbox_pred-coco"
    #The ouput directory and prefix for snapshots
    output_dir = "models/NCTU-lite"
    output_prefix = "v1"    
    #The maximum iterations is controlled in here instead of in solver
    max_iters = 5 * 10000       
    net_params = (solver, train_pt, caffenet,bbox_pred_name, max_iters, output_dir, output_prefix)
    
    GPU_ID=2
    #Start to finetune
    finetune(net_params, roidb,GPU_ID)
    