# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

#This is negative_ignore version of imdb class for Caltech Pedestrian dataset


import re
import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import json
from os import listdir
from os.path import isfile, join
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import glob
import cv2
from datasets.config import CLASS_SETS
from natsort import natsorted


def get_data_map(path="/root/data", prefix="data-"):
    data_map = {} 
    data_paths = glob.glob("{}/{}*".format(path, prefix))
    for data_path in data_paths:
        name = os.path.basename(data_path)[5:]
        data_map[name] = data_path
    return data_map    

data_map = get_data_map()
data_names = data_map.keys()
    
    

def has_data(name):
    return name in data_names

def load_meta(meta_path):
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    else:
       
        meta = {"format":"jpg"}
        meta["train"] = {"start":None, "end":None, "stride":1, "sets":[0]}
        meta["test"] = {"start":None, "end":None, "stride":30, "sets":[1]}
        print("Meta data path: {} does not exist. Use Default meta data: {}".format(meta_path, meta))
    return meta
    
    



class VaticData(imdb):
    def __init__(self, name, class_set_name="pedestrian", train_split="train", test_split="test"):
        
        imdb.__init__(self,'vatic_' + name)
        assert data_map.has_key(name),\
        'The {} dataset does not exist. The available dataset are: {}'.format(name, data_map.keys())
            
        self._data_path = data_map[name]  
        assert os.path.exists(self._data_path), \
        'Path does not exist: {}'.format(self._data_path)
        
        
        self._classes = CLASS_SETS[class_set_name]
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        
        annotation_path = os.path.join(self._data_path, "annotations.json")         
        assert os.path.exists(annotation_path), \
                'Annotation path does not exist.: {}'.format(annotation_path)
        self._annotation = json.load(open(annotation_path))   
        
        
        meta_data_path = os.path.join(self._data_path, "meta.json")         
       
            
        self._meta = load_meta(meta_data_path)
        if train_split == "train" or "test":
            self._train_meta = self._meta[train_split]
        else:
            raise("Options except train and test are not supported!")
            
            
        if test_split == "train" or "test":
            self._train_meta = self._meta[test_split]
        else:
            raise("Options except train and test are not supported!")
        

        self._image_ext = self._meta["format"]    
        self._image_ext = '.jpg'
        self._image_index = self._get_image_index()

       
    


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        set_num, v_num, frame = index.split("_")
        image_path = os.path.join(self._data_path, 'images', set_num, v_num, index+self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
#Strategy: get the index from annotation dictionary 
    
   
    def _load_image_set_list(self):
        image_set_file = os.path.join(self._data_path,
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        f = open(image_set_file)
        return  [line.strip() for line in f]
    
    
       
    def all_index(self, image_set_list):
        image_index = []        
        for set_num in self._annotation:
            if int(set_num[3:]) in image_set_list:
                print("Loading: {}".format(set_num))
                for v_num in self._annotation[set_num]:
                    for frame_num in self._annotation[set_num][v_num]["frames"]:
                        image_index.append("{}_{}_{}".format(set_num, v_num, frame_num))
                     
        return image_index                   
                    

   
        

    def _get_image_index(self):
      
        """
        Load the indexes listed in this dataset's image set file.
        """
        
        
        image_path = os.path.join(self._data_path, 'images')
        assert os.path.exists( image_path), \
                'Path does not exist: {}'.format( image_path)
        target_imgs = []
        
        sets = self._meta["train"]["sets"]
        start = self._meta["train"]["start"]
        end = self._meta["train"]["end"]
        stride = self._meta["train"]["stride"]
        
        
        if start is None:
            start = 0
            
        for set_num in self._meta["train"]["sets"]:
            img_pattern = "{}/set0{}/V000/set0{}_V*.jpg".format(image_path,set_num,set_num)       
            img_paths = natsorted(glob.glob(img_pattern))
            print(img_paths)
            
            first_ind = start
            last_ind = end if end else len(img_paths)
            for i in range(first_ind, last_ind, stride):
                img_path = img_paths[i]
                img_name = os.path.basename(img_path)
                target_imgs.append(img_name[:-4])
               
        print("Total: {} images".format(target_imgs))            
        return target_imgs                  
            
                                
    
                        
        
         
    
    

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions 

        This function loads/saves from/to a cache file to speed up future calls.
        """
  
        
        gt_roidb = []
        for index in self.image_index:
            set_num, v_num, frame = index.split("_")
            #print(set_num)
            set_num = str(int(set_num[-2:]))
            
            
            boxes = self._load_boxes(set_num, frame)
            gt_roidb.append(boxes)

   

        return gt_roidb


    def rpn_roidb(self):
       
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)


        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

  
    
    
    
    #Assign negtaive example to __background__ as whole image
    def _load_boxes(self, set_num ,frame):
     
        bboxes = self._annotation[set_num].get(frame, {}).values()
        #print(frame, "Before", bboxes)
        bboxes = [bbox for bbox in bboxes if bbox['outside']==0 and bbox['occluded']==0]
        if len(bboxes) > 0:
            print("After", bboxes)
        
        num_objs = len(bboxes)
        if num_objs > 0:
            print(bboxes)
            

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        #Becareful about the coordinate format
        # Load object bounding boxes into a data frame.
  
        #print(bboxes)
        # This is possitive example
        for ix, bbox in enumerate(bboxes):
            #print(bbox)
         
          
            x1 = float(bbox['x1'])
            y1 = float(bbox['y1'])
            width = float(bbox['width'])
            height = float(bbox['height'])
            x2 = x1 + width 
            y2 = y1 + height
            label = bbox['label']
            cls = self._class_to_ind[label]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = 1  #Must be pedestrian
            overlaps[ix, cls] = 1.0
            
            
             
            seg_areas[ix] = (x2 - x1) * (y2 - y1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}


    def _get_caltech_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._salt + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path
    # This method write results files into Evaluation toolkit format
    def _write_caltech_results_file(self, net):
         
        #Insert my code in the following space
        
        # The follwing nested fucntions are for smart sorting
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [ atoi(c) for c in re.split('(\d+)', text) ]
        
        def insert_frame(target_frames, file_path,start_frame=29, frame_rate=30):
            file_name = file_path.split("/")[-1]
            set_num, v_num, frame_num = file_name[:-4].split("_")
            if int(frame_num) >= start_frame and int(frame_num) % frame_rate == 29:
                target_frames.setdefault(set_num,{}).setdefault(v_num,[]).append(file_path)
                return 1
            else:
                return 0
       

                    
          
        def detect(file_path,  NMS_THRESH = 0.3):
            im = cv2.imread(file_path)
            scores, boxes = im_detect(net, im)
            cls_scores = scores[: ,1]
            cls_boxes = boxes[:, 4:8]
            dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            return dets[keep, :]
             
        
        def get_target_frames(image_set_list,  image_path):
            target_frames = {}
            total_frames = 0 
            for set_num in image_set_list:
                file_pattern = "{}/set{}_V*".format(image_path,set_num)
                file_list = sorted(glob.glob(file_pattern), key=natural_keys)
                for file_path in file_list:
                    total_frames += insert_frame(target_frames, file_path)
                
            return target_frames, total_frames 
        
        def detection_to_file(target_path, v_num, file_list, detect,total_frames, current_frames, max_proposal=100, thresh=0):
            timer = Timer()
            w = open("{}/{}.txt".format(target_path, v_num), "w")
            for file_index, file_path in enumerate(file_list):
                file_name = file_path.split("/")[-1]
                set_num, v_num, frame_num = file_name[:-4].split("_")
                frame_num = str(int(frame_num) +1)
                
                timer.tic()
                dets = detect(file_path)
               
                timer.toc()
                 
                print('Detection Time:{:.3f}s on {}  {}/{} images'.format(timer.average_time,\
                                                       file_name ,current_frames+file_index+1 , total_frames))
                
                             
                inds = np.where(dets[:, -1] >= thresh)[0]     
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
            
                    x = bbox[0]
                    y = bbox[1] 
                    width = bbox[2] - x 
                    length =  bbox[3] - y
                    w.write("{},{},{},{},{},{}\n".format(frame_num, x, y, width, length, score*100))
                    
               
            w.close()
            print("Evalutaion file {} has been writen".format(w.name))   
            return file_index + 1
               
            
            
                        
        model_name = net.name
        output_path = os.path.join(self._data_path,"res" , self.version, model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)       
            
        
        
        image_set_list = self._load_image_set_list()
   
        
        
        image_path = os.path.join(self._data_path, 'images')
        assert os.path.exists(image_path),'Path does not exist: {}'.format(image_path)
        target_frames, total_frames = get_target_frames(image_set_list,  image_path)
        
       
        
        
        
        
        current_frames = 0
        for set_num in target_frames:
            target_path = os.path.join(output_path, set_num)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            for v_num, file_list in target_frames[set_num].items():
                current_frames += detection_to_file(target_path, v_num, file_list, detect, total_frames, current_frames)
                
       
 
 
#Unit Test
if __name__ == '__main__':
    from datasets.vatic import VaticData
    name = "chruch_street"
    class_set_name = "pedestrian"
    d = VaticData(name, class_set_name)
    #res = d.roidb
    from IPython import embed; embed()