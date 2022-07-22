import cv2
import os
from PIL import Image
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create();
path="train"

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    dic = {}
    i = 0
    for imagePath in imagePaths:
        dic[i] = imagePath
        for Path in [os.path.join(imagePath,f) for f in os.listdir(imagePath)]:
            faceImg=Image.open(Path).convert('L');
            faceNp=np.array(faceImg,'uint8')
            ID=i
            faces.append(faceNp)
            IDs.append(ID)
            cv2.imshow("training",faceNp)
            cv2.waitKey(10)
        i = i + 1
    return np.array(IDs), faces ,dic

Ids, faces, dic = getImagesWithID(path)
print(dic)
recognizer.train(faces, Ids)
recognizer.write('trainningData.yml')
