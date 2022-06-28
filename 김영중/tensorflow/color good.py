#살색만 남기는 코드
import cv2
video = cv2.VideoCapture(0)
cv2.namedWindow('result')
lower_sal = (0,0,0)
upper_sal = (40,255,255)
while 1 :
    ret,img = video.read() #영상을 캠에서 불러옴
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, lower_sal, upper_sal)
    img_result = cv2.bitwise_and(img,img,mask= img_mask)

    if ret == False: # 캠에서 영상이 불러와 지지 않을 경우 반복문 탈출
        break
    cv2.imshow('result', img_result)
    if cv2.waitKey(1) & 0xFF == 27: #esc 누르면 종료
        break
video.release() # video 풀어줌
cv2.destroyAllWindows() # 윈도우 모두 닫기
