import cv2
video = cv2.VideoCapture('vid.avi')

while 1 :
    ret,img = video.read()
    if ret == False:
        break
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
video.release()
cv2.destroyAllWindows()
