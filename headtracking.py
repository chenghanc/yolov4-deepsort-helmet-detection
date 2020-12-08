from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from datetime import datetime
import matplotlib.pyplot as plt
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

FLAGS = {
    'HELMET_DRAW_ENABLED' : False,
    'SAVE_ON_NEW_HEAD' : False,
    'SHOW_ORIGINAL_IMAGE' : True,
    'SHOW_FPS' : True,
    'Tracker_ID' : True
}

def resizeCoord(original_image_shape, network_image_size, coordinate):
    resize_ratio = (original_image_shape[1]/network_image_size[0], original_image_shape[0]/network_image_size[1])

    return int(coordinate[0] * resize_ratio[0]), \
           int(coordinate[1] * resize_ratio[1]), \
           int(coordinate[2] * resize_ratio[0]), \
           int(coordinate[3] * resize_ratio[1])
    

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


netMain = None
metaMain = None
altNames = None

def YOLO(videopath):

    global metaMain, netMain, altNames
    #configPath = "./configs/bgv2.cfg"
    #weightPath = "./configs/bgv2_8000.weights"
    configPath = "./configs/baby-tiny640.cfg"
    weightPath = "./configs/baby-tiny640_best.weights"
    metaPath   = "./configs/baby.data"

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                            print('Object Names = ',altNames)
                except TypeError:
                    pass
        except Exception:
            pass

    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    """
    DeepSORT Parameters
    """
    max_cosine_distance = 0.3  # 0.5 0.4  0.3
    nn_budget = None
    nms_max_overlap = 0.5      #     1.0  0.5

    # load DeepSORT model
    sort_model_file = "model_data/mars-small128.pb"
    encoder = gdet.create_box_encoder(sort_model_file, batch_size=32)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load video file / streams
    cap = cv2.VideoCapture(videopath)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_dimension = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print(original_fps, original_dimension)
    # create head detection result saving directory
    filename = videopath.split(".")[0].split("/")[1]
    directory = os.path.join(os.getcwd(), "outputs", filename)
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    # create video output directory
    out_directory = os.path.join(os.getcwd(), "outputs", "video")
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
        
    # create VideoWriter for output video
    out_write = cv2.VideoWriter( os.path.join(out_directory, filename+"_processed.mp4")
                               , cv2.VideoWriter_fourcc(*'MP4V')
                               , original_fps
                               , original_dimension
                               )
    
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    # network image size (416*416, 512*512, ...)
    network_image_size = (darknet.network_width(netMain),
                          darknet.network_height(netMain))
    print('Network Resolutions = ', network_image_size)
    fps = 0.0

    # head detection id array
    head_set = set()

    while True:
        ret, frame_read = cap.read()
        if ret:
            t1 = time.time()
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       network_image_size,
                                       interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            # get inference information from Yolov4 Model (class, probability, (x,y,width,height))
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
            
            # DeepSORT inference
            bboxes = np.array([x[2] for x in detections])
            scores = np.array([x[1] for x in detections])
            classes = np.array([x[0].decode() for x in detections])
            features = encoder(frame_resized, bboxes)

            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, classes, features)]
            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            # call the tracker
            tracker.predict()
            tracker.update(detections)
            
            # map color to draw random color for each sorting
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # DeepSORT results
            for track in tracker.tracks:
                
                class_name = track.get_class()
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                # if class is 'helmet', ignore
                if not FLAGS['HELMET_DRAW_ENABLED'] and class_name == "helmet":
                    continue

                # DeepSORT results
                bbox = track.to_tlbr()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                # resize bounding box to fit in original image
                xmin, ymin, xmax, ymax = resizeCoord(frame_read.shape, network_image_size, (bbox[0], bbox[1], bbox[2], bbox[3]))
                #xmin = (xmin * 2 - xmax)
                #ymin = (ymin * 2 - ymax)

                # converting back: (x, y) = (xmin, ymin)
                x = xmin
                y = ymin  
                w = xmax - xmin
                h = ymax - ymin
                xmin, ymin, xmax, ymax = convertBack(x, y, w, h)

                # draw class, id on image with opacity
                mask_frame = frame_rgb.copy()
                ALPHA = 1 #0.4
                cv2.rectangle(mask_frame, (xmin, ymin-20), (xmin+(len(class_name)+len(str(track.track_id)))*17, ymin), color, -1)
                #text_color = (255,255,255) if class_name == "helmet" else (0,0,0)
                text_color = (255,255,255)
                cv2.putText(mask_frame, f"{class_name} - {track.track_id}", (xmin, ymin-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)
                frame_rgb = cv2.addWeighted(mask_frame, ALPHA, frame_rgb, 1 - ALPHA, 0)
                
                # draw bounding box
                cv2.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), (255,255,255), 2) # color

                # if new head is appear on image, save image
                if FLAGS['SAVE_ON_NEW_HEAD'] and class_name == 'head' and track.track_id not in head_set:
                    head_set.add(track.track_id)
                    print("new head detected")
                    savePath = os.path.join(os.getcwd(), "outputs", filename, f"{track.track_id}_{datetime.now().strftime('%Y_%m_%d %H_%M_%S')}.jpg")
                    print(savePath)
                    cv2.imwrite(savePath, cv2.hconcat([frame_read, cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)]))

                # if enable flag then print details about each track
                if FLAGS['Tracker_ID']:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (xmin, ymin, xmax, ymax)))

            # Yolo Detection results
            for det in detections:
                bbox = det.to_tlbr()
                xmin, ymin, xmax, ymax = resizeCoord(frame_read.shape, network_image_size, (bbox[0], bbox[1], bbox[2], bbox[3]))
                # converting back: (x, y) = (xmin, ymin)
                x = xmin
                y = ymin  
                w = xmax - xmin
                h = ymax - ymin
                xmin, ymin, xmax, ymax = convertBack(x, y, w, h)
                cv2.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax),(255,0,0), 2)

            # draw fps
            if FLAGS['SHOW_FPS']:
                fps = (fps + (1./(time.time() - t1))) / 2
                cv2.putText(frame_rgb, "FPS: {:.2f}".format(fps), (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            # show in windows
            if FLAGS['SHOW_ORIGINAL_IMAGE']:
                cv2.imshow('Original', frame_read)

            result_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            # show result video
            cv2.imshow('Video', result_frame)
            # save result video
            out_write.write(result_frame)
            # press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break
    cap.release()
    out_write.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # file name goes here
    YOLO("test_videos/test.mp4")
