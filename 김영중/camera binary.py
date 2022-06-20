import cv2
video = cv2.VideoCapture(0)

while 1 :
    ret,img = video.read()
    img_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_bin,img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    if ret_bin == False:
        break
    cv2.imshow('video', img_binary)
    if cv2.waitKey(1) & 0xFF == 27:
        break
video.release()
cv2.destroyAllWindows()



