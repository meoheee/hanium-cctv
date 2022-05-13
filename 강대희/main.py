import cv2

# 가중치 파일 경로
cascade_filename = 'haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)


# 영상 재생
def videoDetector(cam, cascade):
    while True:
        # 캡처 이미지 불러오기
        ret, img = cam.read()
        # 영상 압축
        img = cv2.resize(img, dsize=None, fx=0.75, fy=0.75)
        # 그레이 스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 사진 출력
def imgDetector(img, cascade):
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
