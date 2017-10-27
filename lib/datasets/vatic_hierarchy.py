# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------




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
        print("Meta data path: {} does not exist. Use Default meta data".format(meta_path))
    return meta
 
    
    
class IMDBGroup(imdb):
    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            #print(self.roidb[i]['boxes'])
            #print(boxes[:, 2] - boxes[:, 0])
            if not (boxes[:, 2] >= boxes[:, 0]).all():
                print("Abnormal", oldx1 , oldx2)
                print(i)
           
              
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     'sub_gt_classes' : self.roidb[i]['sub_gt_classes'],
                     'flipped' : True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2
    
    
    def _check_consistency(self):
        
        for dataset in self._datasets[1:]:
            assert self._datasets[0]._classes == dataset._classes, \
            "The class set are inconsistent.  {}/{}  and {}/{}".format(self._datasets[0].name,\
                                                                       self._datasets[0]._classes, dataset.name, dataset._classes)
            assert self._datasets[0]._sub_classes == dataset._sub_classes, \
            "The class set are inconsistent.  {}/{}  and {}/{}".format(self._datasets[0].name,\
                                                                       self._datasets[0]._sub_classes, dataset.name, dataset.sub_classes) 
            
            
            
            
    
    def _get_img_paths(self):
        
        
        img_paths = []
        
        for dataset in self._datasets:
            for i in range(len(dataset._image_index)):
                img_path = dataset.image_path_at(i)
                img_paths.append(img_path)
            
        return img_paths   
            
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]
    
    
    
    def gt_roidb(self):
        
        gt_roidb = []
        for dataset in self._datasets:
            gt_roidb += dataset.gt_roidb()
        return gt_roidb
    
  
    def bbox_status(self, sub_ignores=["empty", "not-target"]):
        status = {"total": 0}
        sub_status = {"total": 0}
        roidb = self.gt_roidb()
        for data in roidb:
            gt_classes = data['gt_classes']
            sub_gt_classes = data['sub_gt_classes']
            status["total"] += len(gt_classes)
            for gt_class in gt_classes:
                #print(gt_class, len( self._classes))
                gt_name = self._datasets[0]._classes[gt_class]
                status[gt_name] = status.get(gt_name, 0)+ 1
                
            for sub_gt_class in sub_gt_classes:
                sub_gt_name = self._datasets[0]._sub_classes[sub_gt_class]
                if sub_gt_name in sub_ignores:
                    continue
                sub_status["total"] += 1   
                sub_status[sub_gt_name] = sub_status.get(sub_gt_name, 0)+ 1    
            
        for class_name in status:
            if class_name == "total":
                continue
            status[class_name] = (status[class_name], status[class_name] / float(status["total"]))
        for sub_class_name in sub_status:
            if sub_class_name == "total":
                continue
            sub_status[sub_class_name] = (sub_status[sub_class_name], sub_status[sub_class_name] / float(sub_status["total"]))    
            
        
        return status, sub_status
    
        

    
    def __init__(self, datasets):
        self._datasets = datasets
        self._check_consistency()
        self._classes = self._datasets[0]._classes
        self._sub_classes = self._datasets[0]._sub_classes
        name = " ".join([dataset.name for dataset in datasets])
        
        imdb.__init__(self,'IMDB Groups:{}'.format(name))
      

        self._image_index = self._get_img_paths()
        
        
        
        



    
class VaticGroup(imdb):
    
    
    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     'sub_gt_classes' : self.roidb[i]['sub_gt_classes'],
                     'flipped' : True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2
    
    def _check_consistency(self):
        for vatic in self._vatics[1:]:
            assert self._vatics[0]._classes == vatic._classes, \
            "The class set are inconsistent.  {}/{}  and {}/{}".format(self._vatics[0].name,\
                                                                       self._vatics[0]._classes, vatic.name, vatic._classes)
    def _get_image_index(self):
        
        
        def restore_path(vatic):
            paths = []
            for img_index in vatic._image_index:
                set_num = img_index[:5]
                path = os.path.join(vatic._data_path, "images", set_num, "V000",img_index)
                paths.append(path)
            return paths
        
        
        target_imgs = []
        for vatic in self._vatics:
            target_imgs += restore_path(vatic)
        return target_imgs    
            
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i] + self._image_ext
    
    
    image_path_at
    def gt_roidb(self):
 
       
        gt_roidb = []
        for vatic in self._vatics:
            for index in vatic._image_index:
                set_num, v_num, frame = index.split("_")
                set_num = str(int(set_num[-2:]))
                boxes = vatic._load_boxes(set_num, frame)
                gt_roidb.append(boxes)
        return gt_roidb


    
    def __init__(self, vatics):
        self._vatics = vatics
        self._check_consistency()
        name = " ".join([vatic.name for vatic in vatics])
        
        imdb.__init__(self,'Vatic Group:{}'.format(name))
         
     

        self._image_ext = '.jpg'
        self._image_index = self._get_image_index()


        
     
        


class VaticData(imdb):
    def __init__(self, name, classes, sub_classes, train_split="train", test_split="test", CLS_mapper={}):
        
        imdb.__init__(self,'vatic_' + name)
        assert data_map.has_key(name),\
        'The {} dataset does not exist. The available dataset are: {}'.format(name, data_map.keys())
            
        self._data_path = data_map[name]  
        assert os.path.exists(self._data_path), \
        'Path does not exist: {}'.format(self._data_path)
        
        self.CLS_mapper = CLS_mapper
        
        self._classes = classes
        self._sub_classes = sub_classes
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._sub_class_to_ind = dict(zip(self._sub_classes, xrange(self.num_classes)))
        
        annotation_path = os.path.join(self._data_path, "annotations.json")         
        assert os.path.exists(annotation_path), \
                'Annotation path does not exist.: {}'.format(annotation_path)
        self._annotation = json.load(open(annotation_path))
        
        self.original_classes = self.get_original_classes()
        
        
        
        
        meta_data_path = os.path.join(self._data_path, "meta.json")         
       
            
        self._meta = load_meta(meta_data_path)
        
        if train_split == "train" or train_split ==  "test":
            pass
        elif train_split == "all":
            print("Use both split for training")
            self._meta["train"]["sets"] +=  self._meta["test"]["sets"]
        else:
            raise("Options except train and test are not supported!")
            
            
        if test_split == "train" or test_split ==  "test":
            pass
        elif test_split == "all":
            print("Use both split for testing")
            self._meta["test"]["sets"] +=  self._meta["train"]["sets"]
        else:
            raise("Options except train and test are not supported!")
        

        self._image_ext = self._meta["format"]    
        self._image_ext = '.jpg'
        self._image_index = self._get_image_index()
        
        
        
    def bbox_status(self, sub_ignores=["empty", "not-target"]):
        status = {"total": 0}
        sub_status = {"total": 0}
        roidb = self.gt_roidb()
        for data in roidb:
            gt_classes = data['gt_classes']
            sub_gt_classes = data['sub_gt_classes']
            status["total"] += len(gt_classes)
            for gt_class in gt_classes:
                #print(gt_class, len( self._classes))
                gt_name = self._classes[gt_class]
                status[gt_name] = status.get(gt_name, 0)+ 1
                
            for sub_gt_class in sub_gt_classes:
                sub_gt_name = self._sub_classes[sub_gt_class]
                if sub_gt_name in sub_ignores:
                    continue
                sub_status["total"] += 1   
                sub_status[sub_gt_name] = sub_status.get(sub_gt_name, 0)+ 1    
            
        for class_name in status:
            if class_name == "total":
                continue
            status[class_name] = (status[class_name], status[class_name] / float(status["total"]))
        for sub_class_name in sub_status:
            if sub_class_name == "total":
                continue
            sub_status[sub_class_name] = (sub_status[sub_class_name], sub_status[sub_class_name] / float(sub_status["total"]))    
            
        
        return status, sub_status    
    
    
    def get_original_classes(self):
        original_classes = set()
        for set_num in self._annotation:
            for bboxes in self._annotation[set_num].values():
                for bbox in bboxes.values():
                    original_classes.add(bbox['label'])
        return original_classes                 
       
    


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
            #print(img_paths)
            
            first_ind = start
            last_ind = end if end else len(img_paths)
            for i in range(first_ind, last_ind, stride):
                img_path = img_paths[i]
                img_name = os.path.basename(img_path)
                target_imgs.append(img_name[:-4])
        print(self._meta)       
        print("Total: {} images".format(len(target_imgs)))            
        return target_imgs                  
                

        
    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     'sub_gt_classes' : self.roidb[i]['sub_gt_classes'],
                     'flipped' : True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

        
        
    
  
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
        
        num_objs = len(bboxes)
       
            

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        sub_gt_classes = np.zeros((num_objs), dtype=np.int32)
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
            sub_label = 'not-target'
            #print(label, self.CLS_mapper)
            if label in self.CLS_mapper:
                if label in self._sub_classes:
                    sub_label = label
                label = self.CLS_mapper[label]
            cls = self._class_to_ind[label]
             
            sub_cls = self._sub_class_to_ind[sub_label]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            #This line is for DEBUG
            #print(sub_cls, sub_label) 
            sub_gt_classes[ix] = sub_cls
            overlaps[ix, cls] = 1.0
            
            
             
            seg_areas[ix] = (x2 - x1) * (y2 - y1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas,
                'sub_gt_classes': sub_gt_classes}


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
    from datasets.vatic_hierarchy import VaticData
    from datasets.vatic import VaticGroup
    classes = CLASS_SETS['coco']
    sub_classes = CLASS_SETS['vehicle-types']
    
    '''
    class_set_name = "pedestrian"
    name_a = "chruch_street"
    
    A = VaticData("chruch_street", class_set_name)
    B = VaticData("YuDa", class_set_name)
    group = VaticGroup([A,B])
    imdb_group = IMDBGroup([A,B])
    '''
    
    mapper = {"van":"car", "trailer-head":"truck",\
              "sedan/suv":"car", "scooter":"motorcycle", "bike":"bicycle"} 
    
    
    A = VaticData("A1HighwayDay", classes, sub_classes, CLS_mapper=mapper)
    B = VaticData("B2HighwayNight", classes, sub_classes, CLS_mapper=mapper)
    group = VaticGroup([A,B])
    imdb_group = IMDBGroup([A,B])
    from IPython import embed; embed()