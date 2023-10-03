import cv2 as cv
import numpy as np

img=cv.imread(r"C:\Users\vivek\OneDrive\Pictures\groupphoto.png")
#cv.imshow('Simba',img)
img=cv.resize(img,(800,800))
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)
harr_cascade=cv.CascadeClassifier('harr_face.xml')
faces_rect=harr_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
print(f'Number of faces found={len(faces_rect)}')

#detecting window

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow("Detected faces",img)

cv.waitKey(0)