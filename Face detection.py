import cv2
import numpy as np

Face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

img = cv2.imread(r"C:\Users\ACER\Desktop\Annotation 2020-06-25 123803.jpg")

faces = Face_cascade.detectMultiScale(img,1.3,5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_color = img[y:y+h,x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_color,2,5)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''1.3 in detectMultiScale() is scale factor as the classifier is trained with 
fixed size of examples our image that is given may not be the correct size so 
1.3 implies zooming that image to that particular factor so then it can detect
and 5 is minneighbors if a face is detected then it is classified as a face only if
there min 5 neighbors to it.'''