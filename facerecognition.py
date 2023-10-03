import cv2 as cv

import numpy as np

haar_cascade=cv.CascadeClassifier('harr_face.xml')
people=['Modi','akhilesh yadav','mamata banarjee','sonia gandi','rahul gandi','mk stalin']
#features=np.load('features.npy',allow_pickle=True)
#labels=np.load('labels.npy',allow_pickle=True)

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img=cv.imread(r"C:\Users\vivek\OneDrive\Pictures\Screenshots\Screenshot 2023-09-21 151229.png")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

faces_rect=haar_cascade.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h,x:x+h]

    label,confidence=face_recognizer.predict(faces_roi)
    print(f'Label={people[label]} with a confidence of {confidence}')

    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detected face',img)
cv.waitKey(0)


