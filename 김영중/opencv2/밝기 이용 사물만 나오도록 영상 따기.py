#밝기를 이용하여 사물만 남기는 코드
import cv2
video = cv2.VideoCapture(0)

cv2.namedWindow('binary') # 두 윈도우 이름 붙여주기
cv2.namedWindow('result')
cv2.createTrackbar('num', 'binary', 0, 255,lambda x : x) #(이름, 윈도우, 범위, 범위, 얼마만큼 올릴래?)바이너리 윈도우에 트래커바 설정
cv2.setTrackbarPos('num', 'binary', 130) #트랙바 초기위치 조정
while 1 :
    ret,img = video.read() #영상을 캠에서 불러옴
    img_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 바이너리로 바꾸기 위해 회색 필터 씌워줌
    divide =  cv2.getTrackbarPos('num', 'binary') # 트랙바에서 값 가져옴
    ret_bin,img_binary = cv2.threshold(img_gray, divide, 255, cv2.THRESH_BINARY_INV) # 이진값으로 바꿔줌 일정 값보다 밝기 작으면 흰색 아니면 검정색
    img_result = cv2.bitwise_and(img, img, mask = img_binary) # 컬러 이미지와 and 연산
    if ret_bin == False: # 캠에서 영상이 불러와 지지 않을 경우 반복문 탈출
        break
    cv2.imshow('binary', img_binary) #영상 보여주기
    cv2.imshow('result', img_result)
    if cv2.waitKey(1) & 0xFF == 27: #esc 누르면 종료
        break
video.release() # video 풀어줌
cv2.destroyAllWindows() # 윈도우 모두 닫기



