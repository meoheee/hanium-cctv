import dlib
import cv2
import time
import numpy as np
import os
import multiprocessing
import threading
import queue

face_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor('/Users/kkjles50/Desktop/동기화된 작업/hanium/pythonProject/shape_predictor_68_face_landmarks.dat')
face_descriptor_extractor = dlib.face_recognition_model_v1('/Users/kkjles50/Desktop/동기화된 작업/hanium/pythonProject/dlib_face_recognition_resnet_model_v1.dat')

video = cv2.VideoCapture(0)
before = time.time()
ret,img = video.read()
face_detection = face_detector(img, 1)
while True:
    ret, img = video.read()
    if time.time() - before > 1:
        face_detection = face_detector(img, 1)
        before = time.time()

    for face in face_detection:
        index = {}
        idx = 0
        face_descriptors = None

        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
        points = points_detector(img, face)
        for point in points.parts():
            cv2.circle(img, (point.x, point.y), 2, (0, 255, 0), 1)

        #68개의 얼굴 특징을 dlib에 보내기 편하도록 변형하는 코드
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(img,points)
        face_descriptor = [f for f in face_descriptor]
        face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
        face_descriptor = face_descriptor[np.newaxis, :]

        #얼굴이 감지되면 이를 하나의 넘파이 어레이에 묶어줌
        if face_descriptors:
            face_descriptors = np.concatenate((face_descriptors, face_descriptor), axis = 0)

    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()