import cv2
import numpy as np

capt = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = capt.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,200,0),2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

capt.release()
cv2.destroyAllWindows()