import cv2
import os

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
imgNum  = 0
name = input('이름 입력  : ')
os.makedirs(f'train/{name}', exist_ok = True)
while True:
    ret, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if ret == False:
        break
    for (x, y, w, h) in faces:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cropped = img[y:y + h, x:x + w]
            cv2.imwrite(f'train/{name}/{imgNum}.png', cropped)
            imgNum += 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break


    cv2.imshow('video', img)

cv2.destroyAllWindows()
