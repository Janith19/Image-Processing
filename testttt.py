import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
import datetime,time
from datetime import datetime
from PyQt5.QtCore import QObject,pyqtSignal
import schedule

from PyQt5.QtWidgets import QApplication, QMainWindow
#import RPi.GPIO as GPIO
import subprocess
#import pyfirmata 
#from pyfirmata import Arduino, OUTPUT, util
import inspect
# from output import Ui_MainWindow
# from set import OtherClass
# GPIO.setmode(GPIO.BOARD)

# Set up GPIO pin for output
# Buzzer = 12
# Light  = 13
# GPIO.setup(Buzzer, GPIO.OUT)
# GPIO.setup(Light, GPIO.OUT)

# def initial_state():
#     GPIO.output(Light, GPIO.LOW)
    

# def crossing_state(t):
#     GPIO.output(Light, GPIO.HIGH)
#     GPIO.output(Buzzer, GPIO.LOW)
#     time.sleep(t)
#     GPIO.output(Light, GPIO.LOW)
#     time.sleep(1)

# def switch_to_initial():
#     GPIO.output(Light, GPIO.LOW)
#     time.sleep(0.5)
    

wait_time_max = 60
person_cnt_max = 2
wait_time_min = 0.8    
'''
board = Arduino('COM16')
  
it = util.Iterator(board)
it.start()    
green = board.get_pin('d:5:p')
red = board.get_pin('d:6:p')
buzzer = board.get_pin('d:9:p')    

'''
import threading
import time
from weather import get_priority_and_weather  # Import the function from the separate file

wait_time_max_default = 3.0  # Default wait time for pedestrians
wait_time_max_vehicle = 2.0  # Adjusted wait time for vehicles

# Initialize wait_time_max outside the function
wait_time_max = wait_time_max_default

# Define a function to adjust wait time based on priority
import threading
import time
from weather import get_priority_and_weather  # Assuming get_priority_and_weather function is correctly implemented

# Define the default and vehicle wait times
wait_time_max_default = 3.0  # Default wait time for pedestrians
wait_time_max_vehicle = 5.0  # Adjusted wait time for vehicles

# Define wait_time_max with an initial value
wait_time_max = wait_time_max_default

# Define a function to adjust wait time based on priority
def adjust_wait_time_based_on_priority():
    global wait_time_max  # Define wait_time_max as global to modify it within the function
    while True:
        latitude = 7.285342
        longitude = 80.726486
        priority, weather_condition = get_priority_and_weather(latitude, longitude) 
        
        # Determine the weather condition and adjust wait time accordingly
        if weather_condition == 'Sunny':
            wait_time_max = wait_time_max_vehicle if priority == 'Vehicles' else wait_time_max_default
        elif weather_condition == 'Rainy':
            wait_time_max = wait_time_max_vehicle if priority == 'Vehicles' else wait_time_max_default

        print("Priority:", priority)
        print("Weather Condition:", weather_condition)
        print("Adjusted wait_time_max:", wait_time_max)
        
        # Sleep for one hour before checking again
        time.sleep(3600)

# Create a separate thread for adjusting wait time based on priority
adjustment_thread = threading.Thread(target=adjust_wait_time_based_on_priority)
adjustment_thread.daemon = True  # Set as daemon so it won't block program exit
adjustment_thread.start()
import threading
import time
from weather import get_priority_and_weather  # Assuming get_priority_and_weather function is correctly implemented

# Define the default and vehicle wait times
wait_time_max_default = 3.0  # Default wait time for pedestrians
wait_time_max_vehicle = 5.0  # Adjusted wait time for vehicles

# Define wait_time_max with an initial value
wait_time_max = wait_time_max_default

# Define a function to adjust wait time based on priority
def adjust_wait_time_based_on_priority():
    global wait_time_max  # Define wait_time_max as global to modify it within the function
    while True:
        latitude = 7.285342
        longitude = 80.726486
        priority, weather_condition = get_priority_and_weather(latitude, longitude) 
        
        # Determine the weather condition and adjust wait time accordingly
        if weather_condition == 'Sunny':
            wait_time_max = wait_time_max_vehicle if priority == 'Vehicles' else wait_time_max_default
        elif weather_condition == 'Rainy':
            wait_time_max = wait_time_max_vehicle if priority == 'Vehicles' else wait_time_max_default

        print("Priority:", priority)
        print("Weather Condition:", weather_condition)
        print("Adjusted wait_time_max:", wait_time_max)
        
        # Sleep for one hour before checking again
        time.sleep(3600)

# Create a separate thread for adjusting wait time based on priority
adjustment_thread = threading.Thread(target=adjust_wait_time_based_on_priority)
adjustment_thread.daemon = True  # Set as daemon so it won't block program exit
adjustment_thread.start()


# Your main code here
# This part of the code will continue to run while the adjustment_thread is also running in the background
while True:
    # Other functionalities of your main code
    time.sleep(1)  # Sleep to avoid high CPU usage



    class PointSelector:
        
        
        
        def __init__(self, video_path):
            self.video_path = video_path
            self.points = []
            self.violation=[]

        def select_points(self):
            
            cap = cv2.VideoCapture(self.video_path)
            _, frame = cap.read()
            target_width = 640
            target_height= 480
            frame = resize_frame(frame,target_width,target_height)
            cv2.putText(frame, f'Point and Click The Waiting Zone and The Violatoin Zone', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.namedWindow("Select Points")
            cv2.imshow("Select Points", frame)
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONUP and len(self.points) < 4:
                    self.points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow("Select Points", frame)
            

            mouse =cv2.setMouseCallback("Select Points", mouse_callback)
            def vio(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONUP and len(self.violation) < 4:
                    self.violation.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    cv2.imshow("Select Points", frame)
            #if (mouse== True):        
            while len(self.points) < 4:

                cv2.waitKey(1)
            cv2.setMouseCallback("Select Points", vio)
            while len(self.violation) < 4:

                cv2.waitKey(1)
            
            cv2.destroyAllWindows()

        def get_points(self):
            return self.points
        def get_vio(self):
            return self.violation


    class PedestrianDetector:
        def __init__(self, weights_path, config_path, names_path, zone_coordinates,violation_cord, video_path):
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            with open(names_path, 'r') as f:
                self.classes = f.read().strip().split('\n')
            
            self.zone_polygon = Polygon(zone_coordinates)
            self.zone_vio = Polygon(violation_cord)
            self.video_path = video_path

        def detect_objects(self, frame):
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
            return outputs

        def run_detection(self):
            
            cap = cv2.VideoCapture(self.video_path)
            start_time = time.time()
            frm_cnt = 0
            total_pedcount = 0
            
            while cap.isOpened():
                
                ret, frame = cap.read()
                frm_cnt += 1
                if 59 > frm_cnt > 31:
                    continue
                if 131 > frm_cnt > 120:
                    continue
                if 176 > frm_cnt > 160:
                    continue
                if 213 > frm_cnt > 190:
                    continue
                if 284 > frm_cnt > 234:
                    continue
                elapsed_time = time.time() - start_time
                
                if frm_cnt == 1:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    durationInSeconds = totalNoFrames // fps
                    durationperframe = durationInSeconds / totalNoFrames
                    print("Total no of Frames:", totalNoFrames)
                    
                    print("Video Duration In Seconds:", durationInSeconds, "s")
                    print("Duration per frame:", durationperframe, "s")
                if not ret:
                    break
                target_width = 640
                target_height= 480
                frame = resize_frame(frame,target_width,target_height)
                outputs = self.detect_objects(frame)
                pedestrian_count = 0
                violation_count = 0
                x_temp =[]
                y_temp =[]
                time_temp = []
                viol = 0 
                temp_dict = {}
                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > 0.5 and self.classes[class_id] == 'head':
                            center_x, center_y, width, height = int(detection[0] * frame.shape[1]), int(detection[1] * frame.shape[0]), \
                                                            int(detection[2] * frame.shape[1]), int(detection[3] * frame.shape[0])
                            x, y = int(center_x - width / 2), int(center_y - height / 2)
                            time_curr = 0
                            time_frm = durationperframe
                            #Predicting and Adjusting missing frames and Keeping the time
                            if frm_cnt > 1:
                                # print("engaged 1st stage")
                                for i in range(len(temp_dict_prev["x"])):
                                    if center_x+10>temp_dict_prev["x"][i]>center_x-10 and center_y+10>temp_dict_prev["y"][i]>center_y-10:
                                        time_curr = time_frm+temp_dict_prev["time"][i]
                                        # print("time_curr1",time_curr)
                                            
                            if frm_cnt > 2 and time_curr==0:
                                # print("engaged 2nd stage")
                                for i in range(len(temp_dict_prev2["x"])):
                                    if center_x+10>temp_dict_prev2["x"][i]>center_x-10 and center_y+10>temp_dict_prev2["y"][i]>center_y-10:
                                        time_curr = time_frm*2+temp_dict_prev2["time"][i]
                                        # print("time_curr2",time_curr)
                                
                            if frm_cnt > 3 and time_curr==0:
                                # print("engaged 3rd stage")
                                for i in range(len(temp_dict_prev3["x"])):
                                    if center_x+10>temp_dict_prev3["x"][i]>center_x-10 and center_y+10>temp_dict_prev3["y"][i]>center_y-10:
                                        time_curr = time_frm*3+temp_dict_prev3["time"][i]
                                        # print("time_curr3",time_curr)
                            

                            if self.zone_polygon.contains(Point(center_x, center_y)):
                                pedestrian_count += 1
                                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)
                                cv2.putText(frame, f'head', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                if time_curr > 0:
                                    time_temp.append(time_curr)
                                    cv2.putText(frame, str(round(time_curr,4))+"s", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 150), 2)
                                else:
                                    time_temp.append(time_frm)
                                    cv2.putText(frame, str(round(time_frm,4))+"s", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 150), 2)
                                
                                x_temp.append(center_x)
                                y_temp.append(center_y)
                                
                                
                                
                            if self.zone_vio.contains(Point(center_x, center_y)):
                                violation_count += 1
                                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 1)
                                cv2.putText(frame, f'Violation', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # if frm_cnt % 30 == 0:
                #     print(f'peestrians detected in the {frm_cnt} frame : {pedestrian_count}')
                #     total_pedcount = total_pedcount+pedestrian_count
                    
                temp_dict["x"] = x_temp
                temp_dict["y"] = y_temp
                temp_dict["time"] = time_temp
                #print("Dict"+str(frm_cnt), temp_dict)
                
                if frm_cnt > 2:
                    temp_dict_prev3 = temp_dict_prev2
                if frm_cnt > 1:
                    temp_dict_prev2 = temp_dict_prev
                temp_dict_prev = temp_dict
                x_temp_prev = x_temp
                y_temp_prev = y_temp
                fps1 = frm_cnt / elapsed_time
                cv2.putText(frame, f'Pedestrians in zone: {pedestrian_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
                #cv2.polylines(frame, [np.array(zone_coordinates, np.int32)], True, (255, 0, 0), 1)
                #cv2.polylines(frame, [np.array(violation_cord, np.int32)], True, (0, 0, 255), 1)
                #Frame Count Text Below
                cv2.putText(frame, f'frames: {frm_cnt}', (target_width-160, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
                cv2.putText(frame, f'Violations Detected: {violation_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
                #cv2.putText(frame, f'fps: {fps}', (1000, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.polylines(frame, [np.array(self.zone_polygon.exterior.coords, np.int32)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [np.array(self.zone_vio.exterior.coords, np.int32)], True, (0, 0, 255), 1)
                cv2.putText(frame, f"FPS: {fps1:.2f}", (target_width-150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
                cv2.imshow('Pedestrian Detection', frame)
                
                # if violation_count >= 1 and GPIO.input(Light) == GPIO.LOW:
                #     GPIO.setmode(GPIO.BOARD)
                #     GPIO.output(Buzzer, GPIO.HIGH)
                #     print("violation detected")
                #     time.sleep(2)
                # else:
                #     GPIO.setmode(GPIO.BOARD)
                #     GPIO.output(Buzzer, GPIO.LOW)
                min_person_cnt = 0
                time_br = 10
                for i in range(len(temp_dict["x"])): 
                    #print(f"person no:{str(i+1)}", "  waiting time:", round(temp_dict["time"][i], 4), "s")
                    if temp_dict["time"][i] > wait_time_min:
                        min_person_cnt += 1
                        #time_br = 10 + (pedestrian_count - 1) * 0.4
                        timebr= 14.14
                        if min_person_cnt>5:
                            time_br=14.14+8
                        elif min_person_cnt>10:
                            time_br=14.14+15
                    if min_person_cnt >= person_cnt_max:
                        #crossing_state(time_br)
                        print("Pedestrian Count is",pedestrian_count-1)
                        append_to_excel_file(pedestrian_count, time.time())
                        
                        #subprocess.Popen(["python", "/home/janith435/Desktop/Gavin/lcd.py"])
                        time.sleep(10)
                        #switch_to_initial()
                
                    if temp_dict["time"][i] >= wait_time_max:
                        print(pedestrian_count-1)
                        append_to_excel_file(pedestrian_count, time.time())
                        #crossing_state(time_br)
                        #switch_to_initial()
                        print("max exceeded")    


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            
            
            cap.release()
            cv2.destroyAllWindows()



    # def append_to_excel_file(pedestrian_count, timestamp):
    #     current_time = datetime.now()
    #     hour_number = current_time.strftime('%H')  # Get the hour number from the current time
    #     human_readable_time = current_time.strftime('%Y-%m-%d %H:%M:%S')  # Convert Unix timestamp to human-readable format
        
    #     data = {'ped_count': [pedestrian_count],
    #             'hour': [hour_number],
    #             'time': [human_readable_time]}
        
    #     try:
    #         # Try reading existing Excel file
    #         df = pd.read_excel('pedcount1.xlsx')
    #         # Append new data to the DataFrame
    #         df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    #     except FileNotFoundError:
    #         # If the file doesn't exist, create a new DataFrame
    #         df = pd.DataFrame(data)
        
    #     # Write DataFrame to Excel file
    #     df.to_excel('pedcount1.xlsx', index=False)
    def append_to_excel_file(pedestrian_count, timestamp):
        current_time = datetime.fromtimestamp(timestamp)
        hour_number = current_time.strftime('%H')  # Get the hour number from the current time
        human_readable_time = current_time.strftime('%Y-%m-%d %H:%M:%S')  # Convert Unix timestamp to human-readable format
        day_of_week = current_time.strftime('%A')  # Get the day of the week

        data = {'ped_count': [pedestrian_count],
                'hour': [hour_number],
                'day_of_week': [day_of_week],
                'time': [human_readable_time]}

        try:
            # Try reading existing Excel file
            df = pd.read_excel('isuri/pedcount_1.xlsx')
            # Append new data to the DataFrame
            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        except FileNotFoundError:
            # If the file doesn't exist, create a new DataFrame
            df = pd.DataFrame(data)

        # Write DataFrame to Excel file
        df.to_excel('isuri/pedcount_1.xlsx', index=False)

    def calculate_average_pedestrians_per_hour():
        # Read the Excel file
        df = pd.read_excel('pedcount1.xlsx')
        
        # Convert 'time' column to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Group by day of the week and hour, then calculate the average pedestrian count
        avg_pedestrians_per_hour = df.groupby(['day_of_week', 'hour'])['ped_count'].mean().reset_index()
        
        # Print the result
        print(avg_pedestrians_per_hour)    

    def resize_frame(frame, target_width, target_height):
        height, width, _ = frame.shape
        aspect_ratio = width / height

        # Calculate new dimensions while maintaining the aspect ratio
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # If the new height exceeds the target height, resize again to the target height
        if new_height > target_height:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
            resized_frame = cv2.resize(frame, (new_width, new_height))

        return resized_frame

    def select_points_and_run_detection(video_path, weights_path, config_path, names_path):
        point_selector = PointSelector(video_path)
        point_selector.select_points()
        detect_vio= point_selector.get_vio()
        selected_points = point_selector.get_points()
        print("Selected Points:", selected_points)
        print("vio points:", detect_vio)
        detector = PedestrianDetector(weights_path, config_path, names_path, selected_points,detect_vio, video_path)
        detector.run_detection()



    # cap = cv2.VideoCapture('C:/Users/Gavin Moragoda/Desktop/Research/interface/0118.mp4')
    # _, frame = cap.read()

    #Paths
    select_points_and_run_detection('D:/Image Processing/Asset1/0118.mp4', 'D:/Image Processing/Asset1/4500weights/custom-yolov4-tiny-detector_best.weights', 'D:/Image Processing/Asset1/4500weights/custom-yolov4-tiny-detector.cfg', 'D:/Image Processing/Asset1/personhead/coco.names')
    #'D:/Image Processing/Asset1/9.mp4'
    #'C:/Users/Gavin Moragoda/Desktop/Research/interface/0118.mp4'
    #'/home/janith435/Desktop/Gavin/0118.mp4'

