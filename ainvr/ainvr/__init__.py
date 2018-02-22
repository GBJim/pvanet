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

cfg_from_file(models_path + "/pvanet/cfgs/submit_160715.yml")

prototxt = models_path + "/pvanet/lite/coco_test.prototxt"
caffemodel = models_path + "/testPedestrian1_iter_130000.caffemodel"
if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found').format(caffemodel_main))



net = caffe.Net(prototxt, caffemodel, caffe.TEST)

caffe.set_mode_gpu()
caffe.set_device(0)
cfg.GPU_ID = 0


