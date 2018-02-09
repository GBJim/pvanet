import skvideo.io
from skvideo.io import FFmpegWriter
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from iou_tracker import Tracker
from util import load_mot
from natsort import natsorted
import os
import glob

def render_detections(im, tracks):
    for id_, track in tracks.items():
        xmin, ymin, xmax, ymax = track
        xmin = int(xmin)
        ymin = int(ymin)
        width = int(xmax - xmin)
        height = int(ymax - ymin)
        label = str(id_)
        highlight_W = xmin + len(label) * 14
        highlight_H = ymin + height
        cv2.rectangle(im, (xmin,ymin),(xmin+width,ymin+height),(0,255,0),2)
        cv2.rectangle(im, (xmin,ymin+height+14),(highlight_W, highlight_H),(0,255,0),-1)
        cv2.putText(im, label, (xmin, highlight_H+14), font, font_size, (0,0,0),1)       
    return im
        

def render_tracks(im, tracks):
    for id_, track in tracks.items():
        xmin, ymin, xmax, ymax = track["bbox"]
        xmin = int(xmin)
        ymin = int(ymin)
        width = int(xmax - xmin)
        height = int(ymax - ymin)
        label = str(id_)
        highlight_W = xmin + len(label) * 14
        highlight_H = ymin + height
        cv2.rectangle(im, (xmin,ymin),(xmin+width,ymin+height),(0,255,0),2)
        cv2.rectangle(im, (xmin,ymin+height+14),(highlight_W, highlight_H),(0,255,0),-1)
        cv2.putText(im, label, (xmin, highlight_H+14), font, font_size, (0,0,0),1)       
    return im
        



def parse_detections(trakcs):
    parsed_detections = []
    for detection in detections:
        score = detection["score"]
        xmin = detection['xmin']
        xmax = detection['xmax'] 
        ymin = detection['ymin']
        ymax = detection['ymax']
        bbox = (xmin, ymin, xmax,  ymax)
        parsed_detections.append({"bbox":bbox, "score":score}) 
       
       
    return parsed_detections



     
def write_video(INPUT_VIDEO, tracks_per_frame, OUTPUT_VIDEO):
    #Reading from images under the given directory
    output_video = FFmpegWriter(OUTPUT_VIDEO)


    if os.path.isdir(INPUT_VIDEO):
        img_paths = natsorted(glob.glob(INPUT_VIDEO+"/*.jpg"))

        for i, img_path in enumerate(img_paths, start=1):
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracks = tracks_per_frame.get(i, {})
            output_frame = render_frame(frame, tracks)
            output_video.writeFrame(output_frame)
            print("Writen Frame: {}".format(i))


    #Reading from a video   
    else:   
        print("Reading Video {}".format(INPUT_VIDEO))
        input_video = skvideo.io.vread(INPUT_VIDEO)
        print("Reading Finished")           
        for i, frame in enumerate(input_video, start=1):      
            tracks = tracks_per_frame.get(i, {})
            output_frame = render_frame(frame, tracks)
            output_video.writeFrame(output_frame)
            print("Writen Frame: {}".format(i))



    output_video.close()
    
    
font = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.8



if __name__ == '__main__':
   
    #INPUT_VIDEO = "./MOT17-04-SDP.mp4"
    INPUT_PATH = "MOT17/train/MOT17-04-SDP/img1"
    INPUT_DETECTION =  "MOT17/train/MOT17-04-SDP/det/det.txt"
    OUTPUT_VIDEO= "./result.mp4"
    T_MAX = 30
    SKIP_RATE = 1
    tracker = Tracker(t_max T_MAX)
    cap = cv2.VideoCapture(INPUT_VIDEO) 
    i = 0 

    while(True):
        ret, frame = cap.read()
        i += 1
        if frame is None:
            break
        if i % SKIP_RATE != 0:
            continue

        detections = ainvr.detect_img(frame)
        parsed_detections = parse_detections(detections)   
        tracks = tracker.track(parsed_detections)   
        frame = render_tracks(frame, tracks) 
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break 


    cap.release() 
    cv2.destroyAllWindows() 
   
