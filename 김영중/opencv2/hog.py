import cv2
import dlib

video = cv2.VideoCapture(0)
facehog = dlib.get_frontal_face_detector()

while 1:
    ret, img = video.read()
    if ret == False:
        break
    detect = facehog(img, 1)
    for i in detect:
        l = i.left()
        t = i.top()
        r = i.right()
        b = i.bottom()
        cv2.rectangle(img,(l,t),(r,b),(0,255))

    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
video.release()
cv2.destroyAllWindows()
