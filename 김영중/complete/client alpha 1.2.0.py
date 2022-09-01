import socket
import cv2
import numpy
import time
import os
import multiprocessing
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video.set(cv2.CAP_PROP_FPS,30)
TCP_IP = 'kkjles50.synology.me'
TCP_PORT = 9505

sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))
print('연결')

def videomake(imgqueue, videoqueue):
    while 1:
        name = 'video/' + str(time.time()) + '.mkv'
        video = cv2.VideoWriter(name,cv2.VideoWriter_fourcc(*'H264'), 30.0, (1280, 720))
        for i in range(30):
            img = imgqueue.get()
            video.write(img)
        video.release()
        videoqueue.put(name)

def send(videoqueue,sock):
    while 1:
        name = videoqueue.get()
        f = open(name,'rb')
        data = f.read()
        length = str(len(data)).ljust(16)
        sock.send(length.encode())
        sock.send(data)
        os.remove(name)

if __name__ == '__main__':
    imgqueue = multiprocessing.Queue()
    videoqueue = multiprocessing.Queue()
    t1 = multiprocessing.Process(target=videomake, args=(imgqueue, videoqueue,))
    t2 = multiprocessing.Process(target=send, args=(videoqueue, sock,))
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()
    while 1 :
        ret,img = video.read()
        if ret == False:
            break
        imgqueue.put(img)
        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    sock.close()
    video.release()
    cv2.destroyAllWindows()