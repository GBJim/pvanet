#!/bin/bash
: '
GPI_ID=1
ITERS=80000
CLASS=coco
DATA=B1StreetDay
TEST_PROTOTXT=models/pvanet/lite/coco_test.prototxt
POSTFIX=80K
'


GPI_ID=1
ITERS=50000
CLASS=coco24
DATA=B1StreetDay
TEST_PROTOTXT=models/pvanet/lite/coco24_test.prototxt
POSTFIX=50K



python tools/test_vatic.py  --data $DATA --output video-coco24-$POSTFIX --weights models/coco/all/coco-24_iter_$ITERS.caffemodel  --net $TEST_PROTOTXT --skip 1 --gpu 0 --class $CLASS

#python tools/test_vatic.py  --data B1StreetDay --output video-all-coco80-10K --weights models/coco/all/all-coco80_iter_10000.caffemodel --net #models/pvanet/lite/coco_test.prototxt --skip 1 --gpu 0 --class coco
#python tools/test_vatic.py  --data B1StreetDay --output video-all-coco80-20K --weights models/coco/all/all-coco80_iter_20000.caffemodel --net #models/pvanet/lite/coco_test.prototxt --skip 1 --gpu 0 --class coco

#python tools/test_vatic.py  --data B1StreetDay --output video-coco80-10K --weights models/coco/all/coco80_iter_10000.caffemodel  --net #models/pvanet/lite/coco_test.prototxt --skip 1 --gpu 0 --class coco
#python tools/test_vatic.py  --data B1StreetDay --output video-coco80-20K --weights models/coco/all/coco80_iter_20000.caffemodel  --net

#models/pvanet/lite/coco_test.prototxt --skip 1 --gpu 0 --class coco
: '
python tools/test_vatic.py  --data $DATA --output video-coco80-$POSTFIX --weights models/coco/all/coco80_iter_$ITERS.caffemodel  --net $TEST_PROTOTXT --skip 1 --gpu 0 --class $CLASS

python tools/test_vatic.py  --data $DATA --output video-balanced-coco80-$POSTFIX --weights models/coco/all/balanced-coco-80_iter_$ITERS.caffemodel  --net $TEST_PROTOTXT --skip 1 --gpu 0 --class $CLASS

python tools/test_vatic.py  --data $DATA --output video-all-coco80-$POSTFIX --weights models/coco/all/all-coco80_iter_$ITERS.caffemodel  --net $TEST_PROTOTXT --skip 1 --gpu 0 --class $CLASS
'