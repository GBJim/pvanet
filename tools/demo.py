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

from ainvr import ainvr
import argparse
import glob
import time

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)', action='store_true')
  
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    if args.cpu_mode:
        ainvr.set_mode_cpu()
    else:
        ainvr.set_mode_gpu(args.gpu_id)
 
    test_imgs = glob.glob("data/demo/*.jpg")
    for test_img in test_imgs:
        tic = time.time()
        outputs = (ainvr.detect(test_img))
        
        toc = time.time()
        print("\n===================")
        print("Image: {} takes {} sec\n".format(test_img, toc-tic))
        print(outputs) 
        print("===================\n")


