import cv2
import numpy as np

# Web camera
cap = cv2.VideoCapture('Resources/res5_video.mp4')

# position of line
count_line_position = 550

# declare minimum width and height of rectangle
min_width_rect = 80 
min_height_rect = 80

# Initialize Substructor
# This algo substract the background from cars in the video because we want to detect cars only
algo = cv2.bgsegm.createBackgroundSubtractorMOG() # algo object gets created of algorithm:- Substractor

# locate center of rectangle
def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

# detects the vehicle and appends in this list
detect = []

# allowable error between pixel
offset = 6

# to count vehicles
counter =0


while True:
    ret,frame1 = cap.read()

    # converting image to grayscale
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Grayscale',grey)

    # to blur the vehicles
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #cv2.imshow('Blurred',blur)

    # applying blur on each frame using algo
    img_sub = algo.apply(blur)
    #cv2.imshow('blur on each frame',img_sub)
    
    # It helps to form the structure of element
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    #cv2.imshow('dilated',dilat)

    # Gives shape to algo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    # gives shape to all the vehicles (changing shape to structure)
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE,kernel)
    #cv2.imshow('Detector',dilatada)

    # count all the vehicles by detecting boundary of vehicles
    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # draw line in video
    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,136,0),3)
    
    # draw rectangle on vehicle
    for(i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_rect) and (h>=min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1, "Vehicle"+str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,200,0),2)

        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)
        
        # count vehicles
        for(x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
                # when vehicle is detected colour of line changes
                cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,136,255),3)
                # now remove the vehicle detected 
                detect.remove((x,y))
                print("vehicle counter:"+str(counter))
    
    
    # Writing on window
    cv2.putText(frame1, " VEHICLE COUNTER : "+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
   
    cv2.imshow('Video Original',frame1)

    # If we press enter then the window will close (13 represents Enter key)
    if cv2.waitKey(1)==13:
        break

# This is used to close the window
cv2.destroyAllWindows()
# This is used to release the video of traffic
cap.release()