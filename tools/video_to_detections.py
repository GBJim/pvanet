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
import skvideo.io
from skvideo.io import FFmpegWriter
import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import glob
from datasets.config import CLASS_SETS


CLASSES = CLASS_SETS['voc']


def detect(im, NMS_THRESH = 0.3, CONF_THRESH=0.75):
    outputs = []    



    #print(img_path)
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    scores, boxes = im_detect(net, im, _t)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = dets[:, -1] >= CONF_THRESH
        dets = dets[keep, :]
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
      
        #inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        
        for i in range(dets.shape[0]):
            bbox = dets[i, :4]
            score = dets[i, -1]
            
            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            output = {"class":cls,"xmin":xmin,"ymin":ymin,"width":xmax - xmin,\
                "height":ymax - ymin, "score":score}
        
            outputs.append(output)

    
    return outputs 




def render_frame(im, dts):
    for dt in dts:
        xmin = dt["xmin"]
        ymin = dt["ymin"]
        width = dt["width"]
        height = dt["height"]
        label = dt['class']
        highlight_W = xmin + len(label) * 14
        highlight_H = ymin + height
        cv2.rectangle(im, (xmin,ymin),(xmin+width,ymin+height),(0,255,0),2)
        cv2.rectangle(im, (xmin,ymin+height+14),(highlight_W, highlight_H),(0,255,0),-1)
        cv2.putText(im, label, (xmin, highlight_H+14), font, font_size, (0,0,0),1)       
    return im
        
        
def write_video(input_path, output_path):
    i = 0
    print("Reading Video {}".format(input_path))
    input_video = skvideo.io.vread(input_path)
    print("Reading Finished")
    output_video = FFmpegWriter(output_path)
    inputs = glob.glob("tools/*.jpg")
    for input_frame in input_video:
        print(input_frame.shape)
        dts = detect(input_frame)
        output_frame = render_frame(input_frame, dts)
        output_video.writeFrame(output_frame)
        i += 1
        print("Writen Frame: {}".format(i))
    output_video.close()
    
    
font = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.8



if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    INPUT_PATH = "/root/data/demo/fruit.mp4"
    OUTPUT_PATH = "/root/data/demo/fruit_result.mp4"
    GPU_ID = 1
    cfg_from_file("models/pvanet/cfgs/submit_1019.yml")
    prototxt = "models/pvanet/lite/coco_test.prototxt"
    caffemodel = "/root/data/PVA-RC/rc3/rc3_iter_200000.caffemodel"



  

    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    cfg.GPU_ID = GPU_ID
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    write_video(INPUT_PATH, OUTPUT_PATH)