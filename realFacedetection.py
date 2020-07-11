# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:39:58 2020

@author: ACER
"""


import cv2


Face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = Face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        reg_of_interest = frame[y:y+h,x:x+h]
        eyes = eye_cascade.detectMultiScale(reg_of_interest,2,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(reg_of_interest,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
print(ret) 