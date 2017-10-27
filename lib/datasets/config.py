CLASS_SETS = {}
CLASS_SETS["pedestrian"] = ('__background__', # always index 0
                         'person')


CLASS_SETS["vehicle-types"] = tuple(['__background__', "van", "sedan/suv", "pickup", "not-target"] + ['empty']* 76)


CLASS_SETS["NCTU-vehicles"] = ('__background__', "van", "sedan/suv", "pickup", 'truck', 'bus',\
                                   'motorcycle', 'bicycle')



CLASS_SETS["3-car"] = ('__background__', "van", "sedan/suv", "pickup")

CLASS_SETS["fire"] = ('__background__', 'fire')


CLASS_SETS["voc"] = ('__background__', # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

CLASS_SETS["coco"] = ('__background__', u'person', u'bicycle', u'car', u'motorcycle', u'airplane', u'bus', u'train', u'truck', u'boat', u'traffic light', u'fire hydrant', u'stop sign', u'parking meter', u'bench', u'bird', u'cat', u'dog', u'horse', u'sheep', u'cow', u'elephant', u'bear', u'zebra', u'giraffe', u'backpack', u'umbrella', u'handbag', u'tie', u'suitcase', u'frisbee', u'skis', u'snowboard', u'sports ball', u'kite', u'baseball bat', u'baseball glove', u'skateboard', u'surfboard', u'tennis racket', u'bottle', u'wine glass', u'cup', u'fork', u'knife', u'spoon', u'bowl', u'banana', u'apple', u'sandwich', u'orange', u'broccoli', u'carrot', u'hot dog', u'pizza', u'donut', u'cake', u'chair', u'couch', u'potted plant', u'bed', u'dining table', u'toilet', u'tv', u'laptop', u'mouse', u'remote', u'keyboard', u'cell phone', u'microwave', u'oven', u'toaster', u'sink', u'refrigerator', u'book', u'clock', u'vase', u'scissors', u'teddy bear', u'hair drier', u'toothbrush')


CLASS_SETS["coco24"] = ('__background__', # always index 0
                         'airplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'dining table', 'dog', 'horse',
                         'motorcycle', 'person', 'potted plant',
                         'sheep', 'couch', 'train', 'tv', 'handbag','backpack','suitcase')
