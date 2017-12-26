"""
COCO 2014 train + coco 2014 val + Vatic[Yuda, A1Highwayday, A2HighwayNight]


Adding NCTU data

"""



"""This is an example of fine tuning PVA-NET through IMDB moudle that combines different dataset together
Most of the codes are modified from Faster R-CNN

"""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import datasets.imdb
from datasets.coco import coco
from datasets.vatic_hierarchy import VaticData, IMDBGroup
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
    main_classes = CLASS_SETS['coco']
    sub_classes = CLASS_SETS['vehicle-types']
    mapper = {"van":"car", "trailer-head":"truck",\
              "sedan/suv":"car", "scooter":"motorcycle", "bike":"bicycle", "pickup": "car"}  
    vatic_names = ["A1HighwayDay", 'B2HighwayNight', "pickup", "tanktruck", "van", "PU_Van", "Sedans_1", "Sedans_2", "van2", "Thailand-long"]
    vatics = [VaticData(vatic_name, main_classes, sub_classes, CLS_mapper=mapper, train_split="all") for vatic_name in vatic_names]   
    #NCTU_VIDEOS = [13, 17, 18, 19, 20, 3, 36, 38, 4, 5 ,6, 7, 8, 9, 10, 11, 12]
    #NCTU_vatic_names = ["NCTU_{}.MOV".format(video) for video in NCTU_VIDEOS]
    #NCTU_vatics = [VaticData(vatic_name, main_classes, sub_classes, CLS_mapper=mapper, train_split="train") for vatic_name in NCTU_vatic_names]   
   
    #imdb_group = IMDBGroup(vatics + NCTU_vatics) 
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
    solver = "models/pvanet/lite/hierachy/solver.prototxt"
    train_pt = "models/pvanet/lite/hierachy/v6_train.prototxt"
    caffenet = "models/rc/rc5/rc5_iter_200000.caffemodel"
    
    #The bbox_pred_name is used to specify the new name of bbox_pred layer in the modified prototxt. bbox_pred layer is handeled differentlly in the snapshooting procedure for the purpose of bbox normalization. In order to prevent sanpshotting the un-tuned bbox_pred layer, we need to specify the new name.  
    bbox_pred_name = "bbox_pred-coco"
    #The ouput directory and prefix for snapshots
    output_dir = "models/hierarchy/v14"
    output_prefix = "v14"    
    #The maximum iterations is controlled in here instead of in solver
    max_iters = 20 * 10000       
    net_params = (solver, train_pt, caffenet,bbox_pred_name, max_iters, output_dir, output_prefix)
    
    GPU_ID=1
    #Start to finetune
    finetune(net_params, roidb,GPU_ID)
    