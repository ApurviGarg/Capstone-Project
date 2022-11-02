import cv2 as cv

cap = cv.VideoCapture('Resources/res3_video.mp4')
cap1 = cv.VideoCapture('Resources/res5_video.mp4')


while (cap.isOpened() or cap1.isOpened()):
   ret, img = cap.read()
   ret1, img1 = cap1.read()
   if ret == True:
       cv.imshow('Video 1',img)
       cv.imshow('Video 2',img1)

   if cv.waitKey(20) and 0xFF == ord('q'):
        break

cap.release()
cap1.release()
cv.destroyAllWindows()