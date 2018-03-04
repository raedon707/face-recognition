import cv2
import numpy as np

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingData.yml")
IDs = 0
font = cv2.FONT_HERSHEY_DUPLEX

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray, 1.5,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        IDs,conf = recognizer.predict(roi_gray)
        cv2.putText(img, str(IDs), (x,y+h), font, 2, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("face", img)
    if(cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllwindows()
