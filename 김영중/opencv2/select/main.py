import cv2
video = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainningData.yml')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
while 1 :
    ret,img = video.read()
    if ret == False:
        break
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = face_detector.detectMultiScale(grey)
    for (x,y,w,h) in detections:
        img_face = cv2.resize(grey[y:(y +w), x:(x+h)],(200,200))
        id, confianca = face_recognizer.predict(img_face)
        print(id)
        name = ''
        if id == 0:
            name = 'kim'
            cv2.putText(img,name,(x,y+w+30),font,2,(0,0,255))

    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
video.release()
cv2.destroyAllWindows()
