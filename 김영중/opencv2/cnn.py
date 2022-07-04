import cv2
import dlib

video = cv2.VideoCapture(0)
facecnn = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

while 1:
    ret, img = video.read()
    if ret == False:
        break
    detect = facecnn(img, 1)
    for i in detect:
        l = i.rect.left()
        t = i.rect.top()
        r = i.rect.right()
        b = i.rect.bottom()
        p = i.confidence
        cv2.rectangle(img,(l,t),(r,b),(0,255))

    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
video.release()
cv2.destroyAllWindows()
