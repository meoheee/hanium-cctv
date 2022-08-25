import socket
import cv2
import dlib
import numpy as np
import os
from PIL import Image

TCP_IP = '192.168.1.3'
TCP_PORT = 9505
face_detector = dlib.cnn_face_detection_model_v1('C:/Users/kkjle/PycharmProjects/pythonProject/mmod_human_face_detector.dat')
face_descriptor_extrator = dlib.face_recognition_model_v1('C:/Users/kkjle/PycharmProjects/pythonProject/dlib_face_recognition_resnet_model_v1.dat')
points_detector = dlib.shape_predictor('C:/Users/kkjle/PycharmProjects/pythonProject/shape_predictor_68_face_landmarks.dat')

#폴더의 사진을 분석하여 학습하는 함수
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
                if face_descriptors is None:
                    face_descriptors = face_descriptor
                else:
                    face_descriptors = np.concatenate((face_descriptors,face_descriptor), axis = 0)
                ids[i] = imagePath[6:]

            print(f'{Path} 사진 분석 완료')
            imgnp = cv2.resize(imgnp, (1280, 720))
            cv2.imshow("training", imgnp)
            cv2.waitKey(100)
            i = i + 1
    cv2.destroyAllWindows()
    return ids, face_descriptors

#학습된 얼굴 데이터를 받아 유사도 계산하는 코드 (이미지, 학습된데이터, 감지범위)
def facerecognize(img, train_descriptors, threshold):
    #얼굴 위치 감지
    detections = face_detector(img, 1)
    for face in detections:
        faceshape = face.rect
        l, t, r, b = (face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom())
        cv2.rectangle(img, (l, t), (r, b), (0, 255, 255), 2)

        #감지된 얼굴의 눈 코 입 형태 감지
        points = points_detector(img, faceshape)
        for point in points.parts():
            cv2.circle(img, (point.x, point.y), 2, (0, 255, 0), 1)
        face_descriptor = face_descriptor_extrator.compute_face_descriptor(img, points)
        distances = np.linalg.norm(train_descriptors - face_descriptor, axis=1)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        if min_distance <= threshold:
            name = ids[min_index]
        else:
            name = 'not indentfied'
        cv2.putText(img, f'{name},{min_distance}', (face.rect.left(), face.rect.top()), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
    return img

#socket 수신 버퍼를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

ids, train_descriptors = facetrain('train')
while 1:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(True)
    conn, addr = s.accept()
    print('연결')
    while 1 :
            # String형의 이미지를 수신받아서 이미지로 변환 하고 화면에 출력
        length = recvall(conn, 16)  # 길이 16의 데이터를 먼저 수신하는 것은 여기에 이미지의 길이를 먼저 받아서 이미지를 받을 때 편리하려고 하는 것이다.
        if not length:
            break
        length.decode()
        stringData = recvall(conn, int(length))
        if not stringData:
            break

        data = np.frombuffer(stringData, dtype='uint8')
        decimg = cv2.imdecode(data, 1)
        decimg = facerecognize(decimg, train_descriptors, 0.38)

        cv2.imshow('video', decimg)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    s.close()
    cv2.destroyAllWindows()


