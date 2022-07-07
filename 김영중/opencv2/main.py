import cv2
import os
from PIL import Image
import numpy as np

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainningData.yml')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
video = cv2.VideoCapture(0)
while (True):
    ret, img = video.read()
    if ret == False:
        break
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = face_detector.detectMultiScale(grey)

    for (x,y,w,h) in detections:
        img_face = cv2.resize(grey[y:(y +w), x:(x+h)],(200,200))
        id, confianca = face_recognizer.predict(img_face)
        print(id)
        name = ''
        if id == 0:
            name = 'kim'
            cv2.putText(img,name,(x,y+w+30),font,2,(0,0,255))
    cv2.imshow('video',img)

camera.release()
cv2.destroyAllWindows()
