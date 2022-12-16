# Import necessary packages

import cv2
import csv
import collections
import numpy as np
from tracker import *
from multiprocessing import Process
import time

# Initialize Tracker
tracker = EuclideanDistTracker()

input_size = 320

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

# Average times for vehicles to pass the intersection
carTime = 2
bikeTime = 1
rickshawTime = 2.25 
busTime = 2.5
truckTime = 2.5
ambulanceTime = 2.5

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


def realTime(video):
        # global timeElapsed
        cap1 = cv2.VideoCapture(video)
        while True:
            success1, img1 = cap1.read()

            if video== 'Resources/res3_video.mp4' or video== 'Resources/res5_video.mp4' :
                img1 = cv2.resize(img1,(0,0),None,0.5,0.5)
            else :
                img1 = cv2.resize(img1,(0,0),None,1,1)
            
            # timeElapsed+=1
            # print("timeelapsed:" , timeElapsed)
            ih1, iw1, channels = img1.shape

            blob1 = cv2.dnn.blobFromImage(img1, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

            # Set the input of the network
            net.setInput(blob1)
          
            
            layersNames = net.getLayerNames()
            outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
            # Feed data to the network
            #print("ouptutNames:", outputNames)
            outputs = net.forward(outputNames)

            # Find the objects from the network output
            postProcess(outputs,img1)
            
            # Draw the crossing lines

            cv2.line(img1, (0, middle_line_position), (iw1, middle_line_position), (255, 0, 255), 2)
            cv2.line(img1, (0, up_line_position), (iw1, up_line_position), (0, 0, 255), 2)
            cv2.line(img1, (0, down_line_position), (iw1, down_line_position), (0, 0, 255), 2)

            # Draw counting texts in the frame
            cv2.putText(img1, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img1, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img1, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img1, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img1, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img1, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img1, "Total:      "+str(up_list[0]+up_list[1]+up_list[2]+up_list[3])+"     "+ str(down_list[0]+down_list[1]+down_list[2]+down_list[3]),(20, 120), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

            # Show the frames
            cv2.imshow('Output1', img1)
          
            if cv2.waitKey(1)==13 :
             break
                

        # Write the vehicle counting information in a file and save it

        with open("data.csv", 'w') as f1:
            cwriter = csv.writer(f1)
            cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
            up_list.insert(0, "Up")
            down_list.insert(0, "Down")
            cwriter.writerow(up_list)
            cwriter.writerow(down_list)
        f1.close()
        # print("Data saved at 'data.csv'")
        # Finally realese the capture object and destroy all active windows
        cap1.release()
         

        cv2.destroyAllWindows()

def Main():
    Videos = ['Resources/res3_video.mp4' , 'Resources/res5_video.mp4' , 'Resources/res6_video.mp4' , 'Resources/res7_video.mp4']
    for i in Videos:
        process = Process(target=realTime, args=(i, ))
        process.start()

if __name__ == '__main__':
    Main()
    #from_static_image(image_file)