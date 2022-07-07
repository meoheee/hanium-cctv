import cv2
import os
from PIL import Image
import numpy as np

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainningData.yml')
camera = cv2.VideoCapture(0)

while (True):
    ret, img = camera.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = face_detector.detectMultiScale(grey, scaleFactor = 1.5, minSize = (30,30))

    for (x,y,w,h)  in detections:
        img_face = cv2.resize(grey,200,200)
        id, confianca = face_recognizer.predict(img_face)
        name = ''
        if id == 0:
            name = 'kim'
        cv2.putText(img,name,(x,y+w+30),2,(0,0,255))
        cv2.putText(img, name, (x, y+h+30), 2, (0, 0, 255))
    cv2.imshow(img)





