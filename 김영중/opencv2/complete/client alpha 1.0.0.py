import socket
import cv2
import dlib
import numpy
import time


video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
TCP_IP = 'kkjles50.synology.me'
TCP_PORT = 9505
FPS = 20

prev_time = 0
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))
print('연결')

while 1 :
    current_time = time.time() - prev_time
    ret,img = video.read()
    if ret == False:
        break
    if (ret is True) and (current_time > 1. / FPS):
        prev_time = time.time()
        
        #추출한 이미지를 String 형태로 변환(인코딩)시키는 과정
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),40]
        result, imgencode = cv2.imencode('.jpg', img, encode_param)
        data = numpy.array(imgencode)
        stringData = data.tobytes()
        length = str(len(stringData)).ljust(16)

        # String 형태로 변환한 이미지를 socket을 통해서 전송
        sock.send(length.encode())
        sock.send(stringData);

        #다시 이미지로 디코딩해서 화면에 출력. 그리고 종료
        decimg=cv2.imdecode(data,1)
        cv2.imshow('video', decimg)
        if cv2.waitKey(1) & 0xFF == 27:
            break
video.release()
cv2.destroyAllWindows()


