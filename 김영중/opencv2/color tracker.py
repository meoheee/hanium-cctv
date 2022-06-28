#살색만 남기는 코드
import cv2
import numpy as np

video = cv2.VideoCapture(0)
cv2.namedWindow('result')
cv2.createTrackbar('threshold', 'result', 0, 255, lambda x : x)
cv2.setTrackbarPos('threshold', 'result', 30)
while 1 :
    #살색만 추출하는 코드
    threshold = cv2.getTrackbarPos('threshold', 'result')
    lower_sal = (0, threshold, 0)
    upper_sal = (30, 255, 255)
    ret,img = video.read() #영상을 캠에서 불러옴
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, lower_sal, upper_sal)

    #모폴로지 연산 - 중간에 꺼멓게 나오는거 제거 함 (노이즈 제거)
    kernel = np.ones((11,11), np.uint8)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)

    img_result = cv2.bitwise_and(img, img, mask=img_mask)

    #물체 위치 추적 코드
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
            cv2.circle(img_result, (centerX, centerY), 10, (0,0,255), 10)
            cv2.rectangle(img_result,(x,y),(x+width,y+width), (0,0,255))


    if ret == False: # 캠에서 영상이 불러와 지지 않을 경우 반복문 탈출
        break
    cv2.imshow('result', img_result)
    if cv2.waitKey(1) & 0xFF == 27: #esc 누르면 종료
        break
video.release() # video 풀어줌
cv2.destroyAllWindows() # 윈도우 모두 닫기





