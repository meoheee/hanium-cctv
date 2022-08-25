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

def receivesocket():


if __name__=='__main__':
    ids, train_descriptors = facetrain('train')
    print(ids)

    imgqueue = multiprocessing.Queue(maxsize=1)
    resultqueue = multiprocessing.Queue()
    t1 = multiprocessing.Process(target=facedetect, args=(imgqueue, resultqueue, ids, train_descriptors,))
    t1.daemon = True
    t1.start()

    TCP_IP = '192.168.1.3'
    TCP_PORT = 9505
    prevtime = 0
    detections = []

    while 1:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(True)
        conn, addr = s.accept()
        print('연결')
        savevideo = cv2.VideoWriter('vid.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1280, 720))
        while 1 :
            currenttime = time.time()

            length = recvall(conn, 16)  # 길이 16의 데이터를 먼저 수신하는 것은 여기에 이미지의 길이를 먼저 받아서 이미지를 받을 때 편리하려고 하는 것이다.
            if not length:
                break
            length.decode()
            stringData = recvall(conn, int(length))
            if not stringData:
                break

            data = np.frombuffer(stringData, dtype='uint8')
            decimg = cv2.imdecode(data, 1)

            if imgqueue.full() == False:
                imgqueue.put(decimg)

            if resultqueue.empty() == False:
                detections = resultqueue.get()

            for result in detections:
                cv2.rectangle(decimg, result[0], result[1], (0, 255, 255), 2)
                for point in result[2].parts():
                    cv2.circle(decimg, (point.x, point.y), 2, (0, 255, 0), 1)
                cv2.putText(decimg, f'{result[4]},{result[3]}', (result[0][0], result[0][1]),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

            sec = currenttime - prevtime
            fps = 1 / sec
            cv2.putText(decimg, f'{fps}', (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            prevtime = currenttime
            cv2.imshow('video', decimg)
            savevideo.write(decimg)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        s.close()
        savevideo.release()
        cv2.destroyAllWindows()
