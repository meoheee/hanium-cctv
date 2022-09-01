import multiprocessing
import socket
import cv2
import dlib
import numpy as np
import time
import os
import threading
import queue
from PIL import Image

face_detector = dlib.cnn_face_detection_model_v1(
    'C:/Users/kkjle/PycharmProjects/pythonProject/mmod_human_face_detector.dat')
face_descriptor_extrator = dlib.face_recognition_model_v1(
    'C:/Users/kkjle/PycharmProjects/pythonProject/dlib_face_recognition_resnet_model_v1.dat')
points_detector = dlib.shape_predictor(
    'C:/Users/kkjle/PycharmProjects/pythonProject/shape_predictor_68_face_landmarks.dat')

def facetrain(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    face_descriptors = None
    ids={}
    i = 0
    for imagePath in imagePaths:
        for Path in [os.path.join(imagePath,f) for f in os.listdir(imagePath)]:
            img = Image.open(Path).convert('RGB')
            imgnp=np.array(img,'uint8')

            detections = face_detector(imgnp, 1)
            for face in detections:
                l, t, r, b = (face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom())
                cv2.rectangle(imgnp, (l, t), (r, b), (0, 255, 255), 2)
                faceshape = face.rect
                points = points_detector(imgnp, faceshape)
                for point in points.parts():
                    cv2.circle(imgnp, (point.x, point.y), 2, (0, 255, 0), 1)
                face_descriptor = face_descriptor_extrator.compute_face_descriptor(imgnp,points)
                face_descriptor = [f for f in face_descriptor]
                face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
                face_descriptor = face_descriptor[np.newaxis, :]
                print(face_descriptor.shape)
                if face_descriptors is None:
                    face_descriptors = face_descriptor
                else:
                    face_descriptors = np.concatenate((face_descriptors,face_descriptor), axis = 0)
                ids[i] = imagePath
            cv2.imshow("training", imgnp)
            cv2.waitKey(100)
            i = i + 1
    cv2.destroyAllWindows()
    return ids, face_descriptors

def facedetect(imgqueue, resultqueue, ids, train_descriptors):

    while 1:
        print('a')
        img = imgqueue.get()
        detections = face_detector(img, 1)
        result = []
        for face in detections:
            l, t, r, b = (face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom())
            points = points_detector(img, face.rect)
            face_descriptor = face_descriptor_extrator.compute_face_descriptor(img, points)
            distances = np.linalg.norm(train_descriptors - face_descriptor, axis=1)
            min_index = np.argmin(distances)
            min_distance = distances[min_index]
            if min_distance <= 0.38:
                name = ids[min_index]
            else:
                name = 'not indentfied'
            result.append([(l,t),(r,b),points,min_distance,name])
        resultqueue.put(result)


#socket 수신 버퍼를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)

    return buf

def recievefile(videoqueue, statusqueue):
    TCP_IP = '192.168.1.3'
    TCP_PORT = 9505
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    while 1:
        s.listen(True)
        conn, addr = s.accept()
        while 1:
            name = 'video/' + str(time.time()) + '.mkv'
            f = open(name, 'wb')
            length = recvall(conn, 16)
            if not length:
                break
            length.decode()
            data = recvall(conn, int(length))
            f.write(data)
            f.close()
            videoqueue.put(name)
            if not data:
                break
        statusqueue.put(0)



if __name__=='__main__':
    ids, train_descriptors = facetrain('train')
    print(ids)

    imgqueue = multiprocessing.Queue(maxsize=1)
    videoqueue = multiprocessing.Queue()
    resultqueue = multiprocessing.Queue()
    statusqueue = multiprocessing.Queue()
    t1 = multiprocessing.Process(target=facedetect, args=(imgqueue, resultqueue, ids, train_descriptors,))
    t1.daemon = True
    t1.start()
    t2 = multiprocessing.Process(target=recievefile, args=(videoqueue,statusqueue))
    t2.daemon = True
    t2.start()


    prevtime = 0
    detections = []
    while 1:
        while 1:
            if statusqueue.empty() == False:
                statusqueue.get()
                break
            name = videoqueue.get()
            video = cv2.VideoCapture(name)
            while 1 :
                ret, img = video.read()
                if ret == False:
                    break
                if imgqueue.full() == False:
                    imgqueue.put(img)

                if resultqueue.empty() == False:
                    detections = resultqueue.get()

                for result in detections:
                    cv2.rectangle(img, result[0], result[1], (0, 255, 255), 2)
                    for point in result[2].parts():
                        cv2.circle(img, (point.x, point.y), 2, (0, 255, 0), 1)
                    cv2.putText(img, f'{result[4]},{result[3]}', (result[0][0], result[0][1]),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))


                cv2.imshow('video', img)
                cv2.waitKey(12)
            video.release()
            os.remove(name)

        cv2.destroyAllWindows()
