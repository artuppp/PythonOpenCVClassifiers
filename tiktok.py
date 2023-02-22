import numpy as np
import cv2

vid = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('C:\\OpenCV\\OpenCV4.6.0G\\data\\haarcascades\\haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('C:\\OpenCV\\OpenCV4.6.0G\\data\\haarcascades\\haarcascade_eye.xml')
mouthCascade = cv2.CascadeClassifier('C:\\OpenCV\\OpenCV4.6.0G\\data\\haarcascades\\haarcascade_smile.xml')

while True:
    ret, frame = vid.read()
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    # Draw just the face
    if len(faces) > 0:
        x, y, w, h = faces[0]
        frame = frame[y:y+h, x:x+w]
        # Resize to original size
        frame = cv2.resize(frame, (640, 480))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()