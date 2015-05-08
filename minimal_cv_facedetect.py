#!/usr/local/bin/ipython
# -*- coding: utf-8 -*-

import cv2
import os

cap = cv2.VideoCapture(0)
cascate_root = '/opt/local/share/OpenCV/haarcascades/'
cascade_name = 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(os.path.join(cascate_root, cascade_name))

while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    faces = face_cascade.detectMultiScale(frame, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
