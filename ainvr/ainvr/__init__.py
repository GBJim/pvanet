import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
caffe_path = osp.join(this_dir, '..', '..', 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

lib_path = osp.join(this_dir, '..', '..', 'lib')
add_path(lib_path)

models_path = osp.join(this_dir, '..', '..', 'models')

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.config import CLASS_SETS
import caffe,os  

CLASSES_main = CLASS_SETS['main']
CLASSES_sub = CLASS_SETS['subordinate'] 

cfg_from_file(models_path + "/pvanet/cfgs/submit_160715.yml")

prototxt_main = models_path + "/pvanet/lite/coco_test.prototxt"
caffemodel_main = models_path + "/rc1_iter_200000.caffemodel"
if not os.path.isfile(caffemodel_main):
        raise IOError(('{:s} not found').format(caffemodel_main))

prototxt_sub = models_path + "/pvanet/lite/coco_test.prototxt"
caffemodel_sub = models_path + "/rc1_iter_200000.caffemodel"
if not os.path.isfile(caffemodel_sub):
        raise IOError(('{:s} not found').format(caffemodel_sub))

net_main = caffe.Net(prototxt_main, caffemodel_main, caffe.TEST)
net_sub = caffe.Net(prototxt_sub, caffemodel_sub, caffe.TEST)

caffe.set_mode_gpu()
caffe.set_device(0)
cfg.GPU_ID = 0


