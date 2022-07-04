import socket
import cv2
import threading

UDP_IP = 'kkjles50.synology.me'
UDP_PORT = 9505

def send_thread(j):
    for k in range(2):
        i=k*10+j
        sock.sendto(bytes([i]) + s[i*46080:(i+1)*46080], (UDP_IP, UDP_PORT))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    d = frame.flatten()
    s = d.tobytes()
    cv2.imshow('video',frame)

    t0 = threading.Thread(target=send_thread, args=[0])
    t1 = threading.Thread(target=send_thread, args=[1])
    t2 = threading.Thread(target=send_thread, args=[2])
    t3 = threading.Thread(target=send_thread, args=[3])
    t4 = threading.Thread(target=send_thread, args=[4])
    t5 = threading.Thread(target=send_thread, args=[5])
    t6 = threading.Thread(target=send_thread, args=[6])
    t7 = threading.Thread(target=send_thread, args=[7])
    t8 = threading.Thread(target=send_thread, args=[8])
    t9 = threading.Thread(target=send_thread, args=[9])
    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    t9.start()
    t0.join()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    t7.join()
    t8.join()
    t9.join()
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release() 
cv2.destroyAllWindows()