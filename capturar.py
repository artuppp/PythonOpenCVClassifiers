import cv2
import numpy as np
import os
import imutils

Datos = 'n'
if not os.path.exists(Datos):
    print('Carpeta creada: ', Datos)
    os.makedirs(Datos)

cap = cv2.VideoCapture(0)

x1, y1 = 190, 80
x2, y2 = 450, 398

count = 0
while True:
    ret, frame = cap.read()
    if ret == False: break
    imAux = frame.copy()
    cv2.rectangle(imAux, (x1, y1), (x2, y2), (0, 255, 0), 2)

    objeto = imAux[y1:y2, x1:x2]
    objeto = imutils.resize(objeto, width=38)

    k = cv2.waitKey(1)
    if k == 27:
        break
    if k == ord('s'):
        cv2.imwrite(Datos + '/objeto_{}.jpg'.format(count), objeto)
        print('Imagen almacenada: ', Datos + '/objeto_{}.jpg'.format(count))
        count = count + 1

    cv2.imshow('frame', imAux)
    cv2.imshow('objeto', objeto)