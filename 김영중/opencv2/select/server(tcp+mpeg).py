import socket
import cv2
import numpy

video = cv2.VideoCapture(0)
#수신에 사용될 내 ip와 내 port번호
TCP_IP = '192.168.1.3'
TCP_PORT = 9505


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
    while 1 :
        ret,img = video.read()
        if ret == False:
            break
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
        cv2.imshow('video', decimg)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    s.close()
    cv2.destroyAllWindows()
video.release()