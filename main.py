import cv2
from ultralytics import YOLO
import math
import supervision as sv
import numpy as np
from supervision import VideoInfo
import time
from datetime import timedelta
import datetime
from pyfirmata import Arduino,util,INPUT,OUTPUT
import time

import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

from invoke import task

@task
def say_hello(c):
    print("Hello, World!")

board = Arduino('COM8')

#generator = sv.get_video_frames_generator('D:/Image Processing/Asset1/bag.mp4')
#iterator = iter(generator)
#frame = next(iterator)


frame_skip = 5
frame_counter = 0

model = YOLO("yolo-Weights/yolov8n.pt")

def RGB(event, x,y,flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x,y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB',RGB)
cap = cv2.VideoCapture('D:/Image Processing/Asset1/bag.mp4')
width = cap.set(3, 1280)
height = cap.set(4, 720)

xax = 366
offset = 6
counter = 0

square = [(454,176),(876,170),(940,356),(411,356)]
square1 = [(335,365),(1013,349),(1183,719),(220,719)]
line = [(940,350),(411,360)]
#line = [(0,373),(1278,347)]


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#video_info = sv.VideoInfo.from_video_path['D:/Image Processing/Asset1/bag.mp4']

polygon = np.array([
        [40 , 40],
        [80,500],
        [600, 400],
        [1200,600]
    ])
#zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    results = model(frame, stream = True)
    #detections= sv.Detections.from_yolov8(results)
    #detections=detections[detections.class_id == 0]
    #zone.trigger(detections = detections)
    out = []
    vio = []
    
    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2),int(y2)
            
            cx=(x1+x2)//2
            cy=(y1+y2)//2
            
            results = cv2.pointPolygonTest(np.array(square,np.int32),(cx,cy),False)
            violation = cv2.pointPolygonTest(np.array(square1,np.int32),(cx,cy),False)
            confidence = math.ceil((box.conf[0]*100))/100
            if results>=0:
                
                if confidence>= float('0.5'):
                    start_time =  time.perf_counter()
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)                   
                    #cv2.circle(frame,(cx,cy),4,(255,0,255),2)                    
                    print(results)
                    print("Confidence --->",confidence)
                    cls = int(box.cls[0])
                    print("Class name --->",classNames[cls])
                    org = [x1,y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color1 = (255,0,0)
                    thickness = 2 
                    elapsed_time = time.perf_counter()-start_time
                    #cv2.putText(frame,str(start_time),org,font,fontScale,color,thickness)
                    cv2.putText(frame,classNames[cls],org,font,fontScale,color1,thickness,) 
                    out.append(classNames[0])
                   # if (xax ==(cy+offset)):
                     #   counter += 1 
                      
            if violation >=1:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)                   
                #cv2.circle(frame,(cx,cy),4,(255,0,255),2)                    
                cls = int(box.cls[0])
                print("Class name --->",classNames[cls])
                org = [x1,y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (0,0,255)
                thickness = 2 
                #cv2.putText(frame,str(start_time),org,font,fontScale,color,thickness)
                cv2.putText(frame,"Violation",org,font,fontScale,color,thickness,) 
                buzzer_pin = 9
                board.digital[buzzer_pin].mode = OUTPUT
                board.digital[buzzer_pin].write(1)   
                 

    vio1 = len(vio)
    buzzer_pin = 9
    board.digital[buzzer_pin].mode = OUTPUT                
    out1 = len(out)   
    if violation>1:
            board.digital[buzzer_pin].write(1)
    else:
        board.digital[buzzer_pin].write(0)
            


    cv2.polylines(frame,[np.array(square1,np.int32)],True,(0,0,255),2)
    cv2.polylines(frame,[np.array(square,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str(out1),(609,656),font,fontScale,color,thickness)
    cv2.imshow('RGB',frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()





