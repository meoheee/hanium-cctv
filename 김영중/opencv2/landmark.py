import socket
import cv2
import numpy as np
import dlib
import os
from PIL import Image

face_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor('/Users/kkjles50/Desktop/동기화된 작업/repository/hanium-cctv/김영중/opencv2/shape_predictor_68_face_landmarks.dat')
face_descriptor_extractor = dlib.face_recognition_model_v1('/Users/kkjles50/Desktop/동기화된 작업/repository/hanium-cctv/김영중/opencv2/dlib_face_recognition_resnet_model_v1.dat')

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    dic = {}
    i = 0
    for imagePath in imagePaths:
        dic[i] = imagePath
        for Path in [os.path.join(imagePath,f) for f in os.listdir(imagePath)]:
            faceImg=Image.open(Path).convert('RGB');
            faceNp=np.array(faceImg,'uint8')
            ID=i
            faces.append(faceNp)
            IDs.append(ID)
            cv2.imshow("training",faceNp)
            cv2.waitKey(10)
        i = i + 1
    return np.array(IDs), faces ,dic

video = cv2.VideoCapture(0)

while 1 :
    ret,img = video.read()
    if ret == False:
        break
    face_detection = face_detector(img,1)
    for face in face_detection:
        l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
        points = points_detector(img, face)
        cv2.rectangle(img, (l,t), (r,b),(0,0,255),2)
        for point in points.parts():
            cv2.circle(img,(point.x,point.y),2,(0,255,0))
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
video.release()
cv2.destroyAllWindows()
