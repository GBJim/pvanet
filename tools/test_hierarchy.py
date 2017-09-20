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
from datasets.factory import get_imdb
from datasets.config import CLASS_SETS
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect_hierarchy
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import glob
import re





def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    
    parser.add_argument('--output', dest='output',  help='model to test', type=str)
    parser.add_argument('--data', dest='data', help='dataset to test', type=str)
    parser.add_argument('--skip', dest='skip', help='skip of the testing', type=int, default=None)
    parser.add_argument('--class', dest='CLS', help='class name to specicied', type=str, default="pedestrian")

    
    
    parser.add_argument('--net', dest='net',
                    help='prototxt file defining the network',
                    default='models/pvanet/full/test.pt', type=str)
    
    
    parser.add_argument('--weights', dest='weights', default='models/pvanet/full/test.model',
                        help='model weights to load', type=str)


    args = parser.parse_args()

    return args





def write_testing_results_file(net, imdb, skip, OUTPUT_DIR, CLASSES):
    
  


    # The follwing nested fucntions are for smart sorting
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split('(\d+)', text) ]
    
    
    

    def insert_frame(target_frames, file_path,start_frame, frame_stride, end_frame):
        file_name = file_path.split("/")[-1]
        set_num, v_num, frame_num = file_name[:-4].split("_")
        condition = int(frame_num) >= start_frame and (int(frame_num)+1) % frame_stride == 0 and int(frame_num) < end_frame
        #print(frame_num,start_frame, frame_stride, end_frame, condition)

        if condition:
            target_frames.setdefault(set_num,{}).setdefault(v_num,[]).append(file_path)
            return 1
        else:
            return 0

 
    
    def detect(img_path):
        
        image_resize = 300 #SSD 500 might need a different size
        net.blobs['data'].reshape(1,3,image_resize,image_resize)
        image = caffe.io.load_image(img_path)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        return net.forward()['detection_out'], image

        #plt.imshow(image)
     


    def get_target_frames(image_set_list, imdb):
        image_path = os.path.join(imdb._data_path, "images")
        start_frame = imdb._meta["test"]["start"]
        end_frame = imdb._meta["test"]["end"]
        frame_stride = skip if skip else imdb._meta["test"]["stride"]
        
        if start_frame is None:
            start_frame = 0
        
        target_frames = {}
        total_frames = 0 
        for set_num in image_set_list:
            file_pattern = "{}/set{}/V000/set{}_V*".format(image_path,set_num,set_num)
            #print(file_pattern)
            #print(file_pattern)
            file_list = sorted(glob.glob(file_pattern), key=natural_keys)
            
            if end_frame is None:
                last_file = file_list[-1]
                end_frame =  int(last_file.split("_")[-1].split(".")[0])
                
            #print(file_list)
            for file_path in file_list:
                total_frames += insert_frame(target_frames, file_path, start_frame, frame_stride, end_frame)

        return target_frames, total_frames 
    
    

    def detection_to_file(target_path, v_num, file_list, detect,total_frames, current_frames, max_proposal=100, thresh=0):
        timer = Timer()
        w = open("{}/{}.txt".format(target_path, v_num), "w")
        for file_index, file_path in enumerate(file_list):
            file_name = file_path.split("/")[-1]
            set_num, v_num, frame_num = file_name[:-4].split("_")
            frame_num = str(int(frame_num) +1)
            im = cv2.imread(file_path)
            timer = Timer()
            timer.tic()
            #print(file_path)
            #print(im.shape)
            #_t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
            _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
            scores,sub_scores, boxes = im_detect_hierarchy(net, im, _t)
            timer.toc()
            print('Detection Time:{:.3f}s on {}  {}/{} images'.format(timer.average_time,\
                                                   file_name ,current_frames+file_index+1 , total_frames))
            
           
            NMS_THRESH = 0.3
            for cls_ind, cls in enumerate(CLASSES[1:]):
                cls_ind += 1 # because we skipped background
                cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                sub_scores_keep = sub_scores[keep,:]
                thresh = 0
                inds = np.where(dets[:, -1] > thresh)[0]   
                
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]*100
                    sub_ind = sub_scores_keep[i].argmax()
                    sub_score = sub_scores_keep[i][sub_ind]
                    sub_cls = sub_classes[sub_ind]

                    #Fix bug 6
                    x = bbox[0]
                    y = bbox[1] 
                    width = bbox[2] - bbox[0] 
                    height =  bbox[3] - bbox[1] 
                    label = cls
                    socre = bbox[-1] * 100
                    w.write("{},{},{},{},{},{},{},{},{}\n".format(frame_num, x, y, width, height, score, label, sub_score, sub_cls))
          

            
            
  

            
        

        w.close()
        print("Evalutaion file {} has been writen".format(w.name))   
            
        
        return file_index + 1
        
        
    image_set_list = [ str(set_num).zfill(2) for set_num in imdb._meta["test"]["sets"]]
    target_frames, total_frames = get_target_frames(image_set_list,  imdb)
    #print(target_frames)
    #print(total_frames)imdb


    current_frames = 0 
    if not os.path.exists(OUTPUT_DIR ):
        os.makedirs(OUTPUT_DIR ) 
    for set_num in target_frames:
        target_path = os.path.join(OUTPUT_DIR , set_num)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for v_num, file_list in target_frames[set_num].items():
            current_frames += detection_to_file(target_path, v_num, file_list, detect, total_frames, current_frames)


    
    

















if __name__ == '__main__':
    
    
    cfg_from_file("models/pvanet/cfgs/submit_1019.yml")
    output_name = "hierarchy"
    data_name = "tanktruck"    
    classes = CLASS_SETS["coco"]
    sub_classes = CLASS_SETS["vehicle-types"]
    prototxt = "models/pvanet/lite/hierachy/v1_test.prototxt"
    caffemodel = "models/hierarchy/v2/v1_iter_10000.caffemodel"
    FPS_rate = 1
    GPU_ID = 0
   


    imdb = get_imdb(data_name, classes)
    
   
    OUTPUT_DIR= os.path.join(imdb._data_path ,"res" ,output_name)    
    if not os.path.exists(OUTPUT_DIR ):
        os.makedirs(OUTPUT_DIR ) 
   
    
  


    if not os.path.isfile(caffemodel):
        raise IOError(('Caffemodel: {:s} not found').format(caffemodel))      
    caffe.set_device(GPU_ID)
    cfg.GPU_ID = GPU_ID
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    
    
    
    print("PVANET Loaded")
    print("Start Detecting")    
    write_testing_results_file(net, imdb, FPS_rate, OUTPUT_DIR, classes)      
