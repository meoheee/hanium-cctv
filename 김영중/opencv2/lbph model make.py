import cv2
import PIL
import dlib
import numpy as np
import os

def get_image():
    path = []
    faces = []
    ids = []
    for i in os.listdir('/Users/kkjles50/Desktop/동기화된 작업/hanium/pythonProject/yalefaces/train'):
        path.append('/Users/kkjles50/Desktop/동기화된 작업/hanium/pythonProject/yalefaces/train/' + i)
    for path in path:
        image = PIL.Image.open(path).convert('L')
        image_np = np.array(image,'uint8')
        id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
        ids.append(id)
        faces.append(image_np)
    return np.array(ids), faces
ids, faces = get_image()
lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces,ids)
lbph_classifier.write('lbph_classifier.yml')