import cv2
import numpy as np
import time

#모폴로지 연산 함수
def morph(img,x=4,y=4):
    kernel = np.ones((4, 4), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

#위치 표시용 함수
def tracker(img,img_mask):
    numoflabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(img_mask)
    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue
        x,y,width,height,area = stats[idx]
        centerX = int(centroid[0])
        centerY = int(centroid[1])
        if area > 300:
            cv2.circle(img, (centerX, centerY), 10, (0,0,255), 10)
            cv2.rectangle(img,(x,y),(x+width,y+width), (0,0,255))

flag = 0
video = cv2.VideoCapture(0)
#원본 파일 저장을 위한 파일 생성
a = cv2.VideoWriter('original.avi',cv2.VideoWriter_fourcc(*'X264'), 24.0, (1280, 720))
#움직임 파악을 위한 객체 생성
fgbg = cv2.createBackgroundSubtractorMOG2(history=None, varThreshold=None, detectShadows=None)

#초기 배경 학습이 안되어 영상 저장되는것을 방지하기 위하여 30프레임은 버림
for i in range(100):
    ret, img = video.read()
    img_mask = fgbg.apply(img)

while 1 :
    ret,img = video.read()
    if ret == False:
        break

    #움직임 파악
    img_mask = fgbg.apply(img)

    #모폴로지 연산 - 노이즈 줄이기
    img_mask = morph(img_mask)

    #움직임 위치 확인
    tracker(img, img_mask)

    #파일 저장 알고리즘?
    a.write(img)
    print(np.any(img_mask>0))
    if np.any(img_mask>0):
        if flag == 0:
            b = cv2.VideoWriter(f'{time.strftime("%Y년%m월%d일 %H:%M:%S")}.avi', cv2.VideoWriter_fourcc(*'X264'), 24.0, (1280, 720))
            flag = 1
        b.write(img)
    elif flag:
        b.release()
        flag = 0

    #현재 영상 확인 및 움직임 파악 영상
    cv2.imshow('video', img)
    cv2.imshow('nobg', img_mask)

    #esc누르믄 탈출
    if cv2.waitKey(1) & 0xFF == 27:
        break
video.release()
a.release()
cv2.destroyAllWindows()