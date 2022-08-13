import cv2
import numpy as np

# Web camera
cap = cv2.VideoCapture('Resources/res3_video.mp4')

# Initialize Substructor




while True:
    ret,frame1 = cap.read()
    cv2.imshow('Video Original',frame1)

    # If we press enter then the window will close
    if cv2.waitKey(1)==13:
        break

# This is used to close the window
cv2.destroyAllWindows()
# This is used to release the video of traffic
cap.release()