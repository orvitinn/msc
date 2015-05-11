#!/usr/local/bin/ipython
# -*- coding: utf-8 -*-

import cv2
import os

cap = cv2.VideoCapture(0)
cascate_root = '/opt/local/share/OpenCV/haarcascades/'
# cascate_root = '/opt/local/share/OpenCV/lbpcascades/'

# cascade_name = 'lbpcascade_frontalface.xml'
cascade_name = 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(os.path.join(cascate_root, cascade_name))

ret, frame = cap.read()
h, w, _ = frame.shape
print w, h


while(True):
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # frame = frame[h/4:h-h/4, 0:w]
    # frame = frame[0:h, w/4:w/4*3]
    faces = face_cascade.detectMultiScale(frame, 1.1, 5, 0, (120, 120), (160, 160))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
