"""
V2: Since V1 has too many False-Positive Examples. We integrate extra data to solve this problem in this stage.
V3: Experiment that increase number of classes will help the feature learning.  Extend to seven classes
V4: Increase the data amount of fire
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
    cfg_path="models/pvanet/lite/train.yml"
    cfg_from_file(cfg_path)
    
    """
     1. PREPARING DATASET
    """

    
    #Fetch the classes of coco dataet, this will be useful in the following section
    #classes = coco_val._classes
    
  
    CLS_mapper = {"bike":"bicycle", 'scooter': "motorcycle", "trailer-head": "truck"}
    
    
    
    
    classes = ('__background__', 'fire', 'person', 'car', 'motorcycle', 'bus', 'truck', 'van', 'bicycle',"pickup", "sedan/suv")
    #Finnaly, let's wrap datasets from Vatic.
    #A vatic dataset directory should be located under ~/data/ directory in the naming of data-*
    #For example: ~/data/data-YuDa,  ~/data/data-A1HighwayDay
    vatic_names = ["chruch_street", "pickup", "A1HighwayDay", "B2HighwayNight", "van", 'PU_Van', "YuDa"]
    vatics = [VaticData(vatic_name, classes, train_split="all",CLS_mapper=CLS_mapper) for vatic_name in vatic_names]
    vatic_fire_1 = VaticData("fire", classes, train_split="all",CLS_mapper=CLS_mapper)
    vatic_fire_2 = VaticData("fire2", classes, train_split="all",CLS_mapper=CLS_mapper)
    vatics.append(vatic_fire_1)
    vatics.append(vatic_fire_2)
    
    
    #Combine all the IMDBs into one single IMDB for training
    datasets = vatics      
    imdb_group = IMDBGroup(datasets)
    imdb, roidb = combined_roidb(imdb_group)
    
    
    total_len = float(len(imdb_group.gt_roidb()))

    #Show the dataset percentage in the whole composition
    for dataset in imdb_group._datasets:
        img_nums = len(dataset.gt_roidb())   
        print(dataset.name, img_nums,  "{0:.2f}%".format(img_nums/total_len * 100))
    
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
    solver = "models/pvanet/lite/ten_solver.prototxt"
    train_pt = "models/pvanet/lite/ten_train.prototxt"
    caffenet = "models/pvanet/lite/test.model"
    
    #The bbox_pred_name is used 'person':"person", to specify the new name of bbox_pred layer in the modified prototxt. bbox_pred layer is handeled differentlly in the snapshooting procedure for the purpose of bbox normalization. In order to prevent sanpshotting the un-tuned bbox_pred layer, we need to specify the new name.  
    bbox_pred_name = "bbox_pred-coco"
    #The ouput directory and prefix for snapshots
    output_dir = "models/fire/v4"
    output_prefix = "v4"    
    #The maximum iterations is controlled in here instead of in solver
    max_iters = 100 * 10000       
    net_params = (solver, train_pt, caffenet,bbox_pred_name, max_iters, output_dir, output_prefix)
    
    GPU_ID = 2
    #Start to finetune
    finetune(net_params, roidb, GPU_ID)
    