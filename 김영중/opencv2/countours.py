#살색만 남기는 코드
import cv2
import numpy as np

video = cv2.VideoCapture(0)
cv2.namedWindow('result')
cv2.namedWindow('binary')
cv2.createTrackbar('threshold1', 'result', 0, 255, lambda x : x)
cv2.setTrackbarPos('threshold1', 'result', 60)
cv2.createTrackbar('threshold2', 'binary', 0, 255, lambda x : x)
cv2.setTrackbarPos('threshold2', 'binary', 60)
while 1 :
    #살색만 추출하는 코드
    threshold1 = cv2.getTrackbarPos('threshold1', 'result')
    lower_sal = (0, threshold1, 0)
    upper_sal = (30, 255, 255)
    ret,img = video.read() #영상을 캠에서 불러옴
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, lower_sal, upper_sal)

    #바이너리 연산
    threshold2 = cv2.getTrackbarPos('threshold2', 'binary')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_bin, img_binary = cv2.threshold(img_gray, threshold2, 255, cv2.THRESH_BINARY)

    #모폴로지 연산 - 노이즈를 제거하기 위한 연산 작업
    kernel = np.ones((11,11), np.uint8)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)

    #컨투어를 이용한 가장자리 따기
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 전체 둘레의 0.05로 오차 범위 지정 ---②
    for i in contours:
        epsilon = 0.005 * cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, epsilon, True)
        cv2.drawContours(img, approx, -1, (255, 0, 0), 10)


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
        if area > 3000:
            cv2.circle(img, (centerX, centerY), 10, (0,0,255), 10)
            cv2.rectangle(img,(x,y),(x+width,y+width), (0,0,255))


    if ret == False: # 캠에서 영상이 불러와 지지 않을 경우 반복문 탈출
        break
    cv2.imshow('result', img_result)
    cv2.imshow('tracker', img)
    cv2.imshow('binary' , img_binary)
    if cv2.waitKey(1) & 0xFF == 27: #esc 누르면 종료
        break
video.release() # video 풀어줌
cv2.destroyAllWindows() # 윈도우 모두 닫기