import cv2

cam = cv2.VideoCapture(0)
a = cv2.VideoWriter('vid.avi',cv2.VideoWriter_fourcc(*'XVID'), 22.0, (1280, 720))
b = cv2.VideoWriter('vid2.avi',cv2.VideoWriter_fourcc(*'XVID'), 22.0, (1280, 720))

while 1 :
    ret,img = cam.read()
    if ret == False:
        continue
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('video', img)
    cv2.imshow('grey', grey)
    a.write(img)
    b.write(grey)
    if cv2.waitKey(1)&0xFF == 27:
        break
cam.release()
a.release()
b.release()
cv2.destroyAllWindows()
