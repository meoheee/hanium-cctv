import cv2
def nothing(x):
    pass
video = cv2.VideoCapture(0)

cv2.namedWindow('binary')
cv2.createTrackbar('num', 'binary', 0, 255, nothing)
cv2.setTrackbarPos('num', 'binary', 127)
while 1 :
    ret,img = video.read()
    img_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    divide =  cv2.getTrackbarPos('num', 'binary')
    ret_bin,img_binary = cv2.threshold(img_gray, divide, 255, cv2.THRESH_BINARY)
    if ret_bin == False:
        break
    cv2.imshow('binary', img_binary)
    if cv2.waitKey(1) & 0xFF == 27:
        break
video.release()
cv2.destroyAllWindows()



