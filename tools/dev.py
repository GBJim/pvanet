#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from datasets.config import CLASS_SETS 
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import glob
import time

CLASSES_main = CLASS_SETS['main']
CLASSES_sub = CLASS_SETS['subordinate'] 

  
    
def detect(net, classes, img_path, CONF_THRESH = 0.8, NMS_THRESH = 0.3):
    outputs = []    
    im = cv2.imread(img_path)
    #print(img_path)
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    scores, boxes = im_detect(net, im, _t)
    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]       
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            output = {"class":cls,"xmin":xmin,"ymin":ymin,"xmax":xmax,\
                "ymax":ymax, "score":score}
            outputs.append(output)

    
    return outputs 
    
def merge_outputs(outputs_main, outputs_sub, IOU_THRESH=0.75):
    
    #Compute IOU tables when two kinds of the outputs are non-zero
    
    if len(outputs_main) and len(outputs_sub):
        for i, box_a in enumerate(outputs_main):
            max_IOU = 0
            max_ID = None
            for j, box_b in enumerate(outputs_sub):
                IOU = get_IOU(box_a, box_b)
                if IOU > max_IOU and IOU > IOU_THRESH:
                    max_IOU = IOU
                    max_ID = j
           
            
            if max_ID is not None:         
                target_box = outputs_sub[max_ID]
                outputs_main[i]['sub'] = {"class": target_box['class'], "score": target_box['score']}
                del outputs_sub[max_ID]
     
                
            else: 
                outputs_main[i]['sub'] = {"class": "no-info", "score":1}
        
        
        
    return outputs_main
    
    
def dual_detect(net_main, net_sub, classes_main, classes_sub, img_path, IOU_THRESH=0.75):
    outputs_main = detect(net_main, CLASSES_main, img_path)
    outputs_sub = detect(net_sub, CLASSES_sub, img_path)
    return  merge_outputs(outputs_main , outputs_sub, IOU_THRESH)
    
 
def get_intersection(box_a, box_b):
    x1_a, y1_a, x2_a, y2_a = box_a["xmin"], box_a["ymin"], box_a["xmax"], box_a["ymax"]
    x1_b, y1_b, x2_b, y2_b = box_b["xmin"], box_b["ymin"], box_b["xmax"], box_b["ymax"]

    #get the width and height of overlap rectangle
    overlap_width =  min(x2_a, x2_b) - max(x1_a, x1_b)
    overlap_height = min(y2_a, y2_b) - max(y1_a, y1_b)

    #If the width or height of overlap rectangle is negative, it implies that two rectangles does not overlap.
    if overlap_width > 0 and overlap_height > 0:
        return overlap_width * overlap_height
    else:
        return 0


    
def get_IOU(box_a, box_b):

    intersection = get_intersection(box_a, box_b)
    if intersection == 0 :
        return 0

    #Union = A + B - I(A&B)
    area_a = (box_a["xmax"] - box_a["xmin"]) * (box_a["ymax"] - box_a["ymin"])
    area_b = (box_b["xmax"] - box_b["xmin"]) * (box_b["ymax"] - box_b["ymin"])
    union_area = area_a + area_b - intersection

    return intersection / union_area    
    
    

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)', action='store_true')
  
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    #Example paramaters configuring 
    cfg_from_file("models/pvanet/cfgs/submit_160715.yml")   
    args = parse_args()
    
    prototxt_main = "models/pvanet/lite/coco_test.prototxt"
    caffemodel_main = "models/rc1_iter_200000.caffemodel"
    if not os.path.isfile(caffemodel_main):
        raise IOError(('{:s} not found').format(caffemodel_main))                    
    
    prototxt_sub = "models/pvanet/lite/coco_test.prototxt"
    caffemodel_sub = "models/rc1_iter_200000.caffemodel"
    if not os.path.isfile(caffemodel_sub):
        raise IOError(('{:s} not found').format(caffemodel_sub))                           
                        
    
    net_main = caffe.Net(prototxt_main, caffemodel_main, caffe.TEST)    
    net_sub = caffe.Net(prototxt_sub, caffemodel_sub, caffe.TEST)
    
    if args.cpu_mode:                    
           caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id     
 
    test_imgs = glob.glob("data/demo/*.jpg")
    for test_img in test_imgs:
        tic = time.time()
        outputs = (dual_detect(net_main, net_sub, CLASSES_main, CLASSES_sub, test_img, IOU_THRESH=0.75))
        
        toc = time.time()
        print("\n===================")
        print("Image: {} takes {} sec\n".format(test_img, toc-tic))
        print(outputs) 
        print("===================\n")