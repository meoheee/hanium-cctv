import cv2
import numpy as np
import time
flag = 0
video = cv2.VideoCapture(0)
a = cv2.VideoWriter('original.avi',cv2.VideoWriter_fourcc(*'X264'), 24.0, (1280, 720))
fgbg = cv2.createBackgroundSubtractorKNN(dist2Threshold = 300,detectShadows=False)
while 1 :
    ret,img = video.read()
    kernel = np.ones((8,8), np.uint8)
    if ret == False:
        break
    img_mask = fgbg.apply(img)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)

    a.write(img)
    print(np.any(img_mask>0))
    if np.any(img_mask>0):
        if flag == 0:
            b = cv2.VideoWriter(f'{time.strftime("%Y년%m월%d일 %H:%M:%S")}.avi', cv2.VideoWriter_fourcc(*'XVID'), 24.0, (1280, 720))
            flag = 1
        b.write(img)
    elif flag:
        b.release()
        flag = 0


    numoflabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(img_mask)
    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue
        x,y,width,height,area = stats[idx]
        centerX = int(centroid[0])
        centerY = int(centroid[1])
        if area > 1000:
            cv2.circle(img, (centerX, centerY), 10, (0,0,255), 10)
            cv2.rectangle(img,(x,y),(x+width,y+width), (0,0,255))

    cv2.imshow('video', img)
    cv2.imshow('nobg', img_mask)
    if cv2.waitKey(1) & 0xFF == 27:
        break
video.release()
a.release()
cv2.destroyAllWindows()