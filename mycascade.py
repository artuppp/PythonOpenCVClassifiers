import numpy as np
import cv2

vid = cv2.VideoCapture(0)
teddyCascade = cv2.CascadeClassifier('classifier\\cascade.xml')

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    teddys = teddyCascade.detectMultiScale(gray, scaleFactor=4, minNeighbors=85, minSize=(70,70))
    # draw rectangle around the teddy
    for (x, y, w, h) in teddys:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()