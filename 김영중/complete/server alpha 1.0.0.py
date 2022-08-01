import socket
import cv2
import dlib
import numpy

#수신에 사용될 내 ip와 내 port번호
TCP_IP = '192.168.1.3'
TCP_PORT = 9505
face_detector = dlib.cnn_face_detection_model_v1('C:/Users/kkjle/PycharmProjects/pythonProject/mmod_human_face_detector.dat')
face_descriptor_extrator = dlib.face_recognition_model_v1('C:/Users/kkjle/PycharmProjects/pythonProject/dlib_face_recognition_resnet_model_v1.dat')
points_detector = dlib.shape_predictor('C:/Users/kkjle/PycharmProjects/pythonProject/shape_predictor_68_face_landmarks.dat')

#socket 수신 버퍼를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

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
        data = numpy.frombuffer(stringData, dtype='uint8')
        decimg = cv2.imdecode(data, 1)
        detections = face_detector(decimg, 1)

        for face in detections:
            faceshape = face.rect
            points = points_detector(decimg, faceshape)
            for point in points.parts():
                cv2.circle(decimg, (point.x,point.y), 2, (0,255,0),1)
            l, t, r, b = (face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom())
            cv2.rectangle(decimg,(l,t),(r,b),(0,255,255),2)
        cv2.imshow('video', decimg)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    s.close()
    cv2.destroyAllWindows()
