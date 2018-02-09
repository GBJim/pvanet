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

from  __init__ import net, CLASSES_main, CLASSES_sub

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect_hierarchy
from fast_rcnn.nms_wrapper import nms
from datasets.config import CLASS_SETS 
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2



def get_sub_cls(scores):
    ind = np.argmax(scores)
    score = scores[ind]
    label = CLASSES_sub[ind]
    return label, score
    

def detect_img(im, roi=(0,0,0,0), NMS_THRESH = 0.3, CONF_THRESH=0.75):
    outputs = []
   

    if roi[2] > 0:
        roiImage = im[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] # y1, y2, x1, x2
    else:
        roiImage = im.copy()

    #print(img_path) 
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    scores, sub_scores, boxes = im_detect_hierarchy(net, roiImage, _t)
    for cls_ind, cls in enumerate(CLASSES_main[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = dets[:, -1] >= CONF_THRESH
        dets = dets[keep, :]
        filtered_sub_scores = sub_scores[keep, :]
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        filtered_sub_scores =  filtered_sub_scores[keep, :]
        #inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            score = dets[i, -1]

            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)

            output = {"class":cls,"xmin":xmin+roi[0],"ymin":ymin+roi[1],"xmax":xmax+roi[0],\
                "ymax":ymax+roi[1], "score":score}

            if cls == "car":
                sub_cls, sub_score = get_sub_cls(filtered_sub_scores[i])
                if sub_cls != "__background__":
                    output["sub"] = {"class": sub_cls, "score":sub_score}



            outputs.append(output)
    return outputs

def detect(img_path, roi=(0,0,0,0), NMS_THRESH = 0.3, CONF_THRESH=0.75):
    outputs = []    
    im = cv2.imread(img_path)

    if roi[2] > 0:
        roiImage = im[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] # y1, y2, x1, x2
    else:
        roiImage = im.copy()

    #print(img_path)
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    scores, sub_scores, boxes = im_detect_hierarchy(net, roiImage, _t)
    for cls_ind, cls in enumerate(CLASSES_main[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = dets[:, -1] >= CONF_THRESH
        dets = dets[keep, :]
        filtered_sub_scores = sub_scores[keep, :]
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        filtered_sub_scores =  filtered_sub_scores[keep, :]
        #inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        
        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            score = dets[i, -1]
            
            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)

            output = {"class":cls,"xmin":xmin+roi[0],"ymin":ymin+roi[1],"xmax":xmax+roi[0],\
                "ymax":ymax+roi[1], "score":score}
           
            if cls == "car":    
                sub_cls, sub_score = get_sub_cls(filtered_sub_scores[i])
                if sub_cls != "__background__":
                    output["sub"] = {"class": sub_cls, "score":sub_score}
                
            
            
            outputs.append(output)

    
    return outputs 

def set_mode_cpu():
    caffe.set_mode_cpu()

def set_mode_gpu(gpu_id=0):
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    cfg.GPU_ID = gpu_id
