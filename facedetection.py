import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

name = raw_input("enter name of user: ")
user_id = raw_input("enter user id: ")
sample_number = 0

cap = cv2.VideoCapture(0)
while True:
    rec, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        cv2.imwrite("dataset/"+str(name)+"."+str(user_id)+"."+str(sample_number)+".jpg", roi_gray)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        #roi_color = img[y:y+h, x:x+w]
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        sample_number += 1
        cv2.waitKey(100);

    cv2.imshow('img', img);
    cv2.waitKey(1);
    print str(sample_number)
    #k = cv2.waitKey(30) & 0xff
    #if k == 27:
    if sample_number > 20:
        break

cap.release()
cv2.destroyAllWindows()
        
