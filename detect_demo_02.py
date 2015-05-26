#!/usr/local/bin/ipython
# -*- coding: utf-8 -*-

import cv2
import os

cap = cv2.VideoCapture(0)
cascate_root = '/opt/local/share/OpenCV/haarcascades/'
cascade_name = 'haarcascade_frontalface_alt.xml'
smile_name = 'haarcascade_smile.xml'
eye_name = 'haarcascade_mcs_eyepair_big.xml'

face_cascade = cv2.CascadeClassifier(os.path.join(cascate_root, cascade_name))
smile_cascade = cv2.CascadeClassifier(os.path.join(cascate_root, smile_name))
eye_pair_cascade = cv2.CascadeClassifier(os.path.join(cascate_root, eye_name))

while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    faces = face_cascade.detectMultiScale(frame, 1.1, 5, 0, (120, 120), (250, 250))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        upper_face = frame[y:y+h/2, x:x+w]
        eyes = eye_pair_cascade.detectMultiScale(upper_face, 1.1, 5, 0, (50, 20))
        for x2, y2, w2, h2 in eyes:
            cv2.rectangle(frame, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (0, 255, 0), 1)

        lower_face = frame[y+h/2:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(lower_face, 1.1, 5, 0, (50, 20), (200, 200))
        for x2, y2, w2, h2 in smile:
            cv2.rectangle(frame, (x+x2, y+h/2+y2), (x+x2+w2, y+h/2+y2+h2), (0, 255, 0), 1)

    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
