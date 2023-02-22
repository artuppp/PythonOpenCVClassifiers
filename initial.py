import numpy as np
import cv2

vid = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('C:\\OpenCV\\OpenCV4.6.0G\\data\\haarcascades\\haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('C:\\OpenCV\\OpenCV4.6.0G\\data\\haarcascades\\haarcascade_eye.xml')
mouthCascade = cv2.CascadeClassifier('C:\\OpenCV\\OpenCV4.6.0G\\data\\haarcascades\\haarcascade_smile.xml')

tm = cv2.TickMeter()
while True:
    ret, frame = vid.read()
    frameCopy = frame.copy()

    tm.start()
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    eyes = eyeCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(15,15), flags=cv2.CASCADE_SCALE_IMAGE)
    tm.stop()

    #mouth = mouthCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(15,15), flags=cv2.CASCADE_SCALE_IMAGE)
    # draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # draw rectangle around the eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # draw rectangle around the mouth
    #for (x, y, w, h) in mouth:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)

    if k & 0xFF == ord('g'):
        cv2.imwrite('rectangulos.jpg', frame)
        cv2.imwrite('original.jpg', frameCopy)

    if k & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()