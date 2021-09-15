import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
from math import atan2
from math import degrees
from gpiozero import Robot
degs=0
rads=0
robot = Robot(left=(20,26), right=(18, 27))
rf= Robot(left=(8,7), right=(16,19))

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        i, j = (x + (x+w)) // 2, (y +(y+h)) // 2
    # Draw a circle in the center of rectangle
        cv2.circle(frame, center=(i,j), radius=3, color=(0,0, 255), thickness=1)
        dy = 360-i
        dx= 640-j
        rads = atan2(dy,dx)
        degs=0
        degs = degrees(rads)
        if degs < 0 :
            degs +=90

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
        if((4.99>degs>0.0)or(degs>=85)):
            robot.forward(0.25)
            rf.forward(0.25)
            print("FORWARD")
        elif((50.0>degs)and(degs>=5.0)):
            robot.left(0.25)
            rf.left(0.25)
            print("LEFT")
        elif((50.0<degs)and (degs<=85.0)):
            robot.right(0.25)
            rf.right(0.25)
            print("RIGHT")
    else:
        robot.left(0.25)
        sleep(1)
        robot.stop()
        rf.stop()
        print("Find Mode")



    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
