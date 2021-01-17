import cv2
from random import randrange
import numpy as np

trained_face_data = cv2.CascadeClassifier('face_detection/haarcascade_frontalface_default.xml')

#for images

ori_img = cv2.imread('face_blur/mk.jpg')

img = cv2.resize(ori_img, (840,500)) 
grayed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayed_image)

for x, y, w, h in face_coordinates:
    ROI = img[y:y+h, x:x+w]
    image = cv2.blur(ROI, (20,20))
    img[y:y+h, x:x+w] = image

cv2.imshow('blurring face', img)
cv2.waitKey()

#for webcam

webcam = cv2.VideoCapture(0)

while True:

    happening, frame = webcam.read()
    grayed_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayed_image)

    for x, y, w, h in face_coordinates:
        ROI = frame[y:y+h, x:x+w]
        image = cv2.blur(ROI, (50,50))
        frame[y:y+h, x:x+w] = image

    cv2.imshow('blurring face', frame)
    key = cv2.waitKey(1)

    if key==113 or key==81:
        break

webcam.release()


print ("code completed")