#!/bin/bash



python tools/test_vatic.py  --data B1StreetDay --output video-all-coco80 --weights models/coco/pvanet/all-coco80_10K.caffemodel  --net models/pvanet/lite/coco_test.prototxt --skip 1 --gpu 3 --class coco