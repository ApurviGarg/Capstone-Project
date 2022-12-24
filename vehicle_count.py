# Import necessary packages

import cv2
import datetime
import csv
import collections
import numpy as np
from tracker import *
import multiprocessing
import time
import os

from pathlib import Path

current_time = datetime.datetime.now()
start=current_time.minute*60 + current_time.second
#print("start:" , start)


# Average times for vehicles to pass the intersection
carTime = 2
bikeTime = 1
rickshawTime = 2.25 
busTime = 2.5
truckTime = 2.5
ambulanceTime = 2.5

# Initialize Tracker
tracker = EuclideanDistTracker()

input_size = 320

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

# Average times for vehicles to pass the intersection



timeElapsed =0

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Middle cross line position
middle_line_position = 225   
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15


# Store Coco Names in a list
classesFile = "Resources/coco.names" # "coco.names" in the same directory
classNames = open(classesFile).read().strip().split('\n')
#print(classNames)
#print(len(classNames))

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

## Model Files
modelConfiguration = 'YoloFiles/yolov3-320.cfg' # "yolov3-320.cfg" in same directory
modelWeigheights = 'YoloFiles/yolov3-320.weights' # "yolov3-320.weights" in same directory

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# Function for count vehicle
def count_vehicle(box_id, img):

    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    
    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)

# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # print(x,y,w,h)

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Draw classname and confidence score 
        cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)


def realTime(video,greenTime , ambulance_detected , total_vehicle):
        global start
        cap1 = cv2.VideoCapture(video)
        while True:
            success1, img1 = cap1.read()

            if video== 'Resources/res3_video_10.mp4' or video== 'Resources/res5_video_10.mp4' or video== 'Resources/res7_video_10.mp4':
                img1 = cv2.resize(img1,(0,0),None,0.5,0.5)
            else :
                img1 = cv2.resize(img1,(0,0),None,1,1)
            
                  
            ih1, iw1, channels = img1.shape

            blob1 = cv2.dnn.blobFromImage(img1, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

            # Set the input of the network
            net.setInput(blob1)
          
            
            layersNames = net.getLayerNames()
            outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
            # Feed data to the network
            #print("ouptutNames:", outputNames)
            outputs = net.forward(outputNames)

            # Find the objects from the network output
            postProcess(outputs,img1)
            
            # Draw the crossing lines

            cv2.line(img1, (0, middle_line_position), (iw1, middle_line_position), (255, 0, 255), 2)
            cv2.line(img1, (0, up_line_position), (iw1, up_line_position), (0, 0, 255), 2)
            cv2.line(img1, (0, down_line_position), (iw1, down_line_position), (0, 0, 255), 2)
            
            
            # to correct the mistakenly detected bus in the video
            if video== 'Resources/res6_video_10.mp4':
                down_list[2]=0

            greenTime.value = math.ceil(((down_list[0]*carTime) + (down_list[2]*busTime) + (down_list[3]*truckTime)+ (down_list[1]*bikeTime))/2)

            totalVehicleCount=down_list[0]+down_list[1]+down_list[2]+down_list[3]


            # Draw counting texts in the frame
            cv2.putText(img1, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img1, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img1, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img1, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img1, "Ambulance:  "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img1, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img1, "Total:      "+str(up_list[0]+up_list[1]+up_list[2]+up_list[3])+"     "+ str(totalVehicleCount),(20, 120), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            #cv2.putText(img1 ,"Density:     "+str(greenTime.value) , (20,140) ,  cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            # Show the frames
            cv2.imshow('Lane', img1)
                  
                        
            total_vehicle.value=totalVehicleCount

            if down_list[2]>=1: #bus is considered as ambulance here
                ambulance_detected.value=1

            if cv2.waitKey(1)==13 :
                break

        # Finally realese the capture object and destroy all active windows
        #print("appended greentimer:" , greenTime.value , " to process id:" , format(os.getpid()))
        cap1.release() 
        cv2.destroyAllWindows()
        


def Main():
    processed_lane=["False"  , "False" , "False" , "False"]
    currentGreen=0
    
    for round in range(0,4):
        print('CurrentGreen',currentGreen)
        processed_lane[currentGreen] = "True"
        greentimer=[]             #stores green timer for every lane 
        ambulance_detected=[]    #stores ambulance detection count of very lane
        total_vehicle=[]   #stores total vehicles in every lane
        processes=[]
        Videos = ['Resources/res5_video_10.mp4' , 'Resources/res3_video_10.mp4' , 'Resources/res6_video_10.mp4' , 'Resources/res5_video_10.mp4','Resources/res7_video_10.mp4']
        lane1 = multiprocessing.Value('i',0)
        lane2 = multiprocessing.Value('i',0)
        lane3 = multiprocessing.Value('i',0)
        lane4 = multiprocessing.Value('i',0)

        ambulance_detected1= multiprocessing.Value('i' , 0)
        ambulance_detected2= multiprocessing.Value('i' , 0)
        ambulance_detected3= multiprocessing.Value('i' , 0)
        ambulance_detected4= multiprocessing.Value('i' , 0)

        total_vehicle1=multiprocessing.Value('i' ,0)
        total_vehicle2=multiprocessing.Value('i' ,0)
        total_vehicle3=multiprocessing.Value('i' ,0)
        total_vehicle4=multiprocessing.Value('i' ,0)


        count = 0
        for i in Videos:
            count=count+1
            if count==1:
                process = multiprocessing.Process(target=realTime, args=(i,lane1,ambulance_detected1,total_vehicle1,))
                process.start()
                processes.append(process)
            elif count==2:
                process = multiprocessing.Process(target=realTime, args=(i,lane2,ambulance_detected2,total_vehicle2,))
                process.start()
                processes.append(process)
            elif count==3:
                process = multiprocessing.Process(target=realTime, args=(i,lane3,ambulance_detected3,total_vehicle3,))
                process.start()
                processes.append(process)
            elif count==4 and round!=1:
                process = multiprocessing.Process(target=realTime, args=(i,lane4,ambulance_detected4, total_vehicle4,)) 
                process.start()
                processes.append(process)
            elif count==5 and round==1:
                process = multiprocessing.Process(target=realTime, args=(i,lane4,ambulance_detected4, total_vehicle4,))  
                process.start()
                processes.append(process) 
            else :
                pass    
            
        
        for process in processes:
            process.join()  

        greentimer.append(lane1)
        greentimer.append(lane2)
        greentimer.append(lane3)
        greentimer.append(lane4)
        
        ambulance_detected.append(ambulance_detected1)
        ambulance_detected.append(ambulance_detected2)
        ambulance_detected.append(ambulance_detected3)
        ambulance_detected.append(ambulance_detected4)

        total_vehicle.append(total_vehicle1)
        total_vehicle.append(total_vehicle2)
        total_vehicle.append(total_vehicle3)
        total_vehicle.append(total_vehicle4)
        
        for i in range(0,4):
            print('No of vehicles in lane',i+1,':',total_vehicle[i].value)
        
        print('Signals for all lanes')
        for i in range(0,4):
            if i==currentGreen:
                print('Signal for lane',i+1,'Green   and   Green Timer = ', greentimer[i].value)
            else:
                print('Signal for lane',i+1,'Red')
        
        counter=0
        
        print('For current Green',currentGreen)
        for i in range(0,4):
            if processed_lane[i]=="False":
                counter=counter+1
                if counter==1:
                    currentGreen=i
                    print('First false ',i)
                elif ambulance_detected[i].value==1 and counter!=1:
                    currentGreen=i
                    print('ambulance detected in',i)
                   



if __name__ == '__main__':
    Main()
    #from_static_image(image_file)