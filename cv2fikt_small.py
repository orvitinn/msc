#!/usr/local/bin/ipython
# -*- coding: utf-8 -*-

import cv2
# import utils
import os
import requests
import thread
import utils
import base64
import json
import time
from Queue import Queue

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

cascate_root = '/opt/local/share/OpenCV/haarcascades/'

face_cascade2 = cv2.CascadeClassifier(os.path.join(cascate_root, 'haarcascade_frontalface_alt.xml'))
face_cascade3 = cv2.CascadeClassifier(os.path.join(cascate_root, 'haarcascade_frontalface_alt2.xml'))
# face_cascade4 = cv2.CascadeClassifier(os.path.join(cascate_root, 'haarcascade_frontalface_alt_tree.xml'))
eye_cascade1 = cv2.CascadeClassifier(os.path.join(cascate_root, 'haarcascade_eye_tree_eyeglasses.xml'))

# smile_cascade = cv2.CascadeClassifier(os.path.join(cascate_root, 'haarcascade_smile.xml'))
# lear_cascade = cv2.CascadeClassifier(os.path.join(cascate_root, 'haarcascade_mcs_leftear.xml'))
# rear_cascade = cv2.CascadeClassifier(os.path.join(cascate_root, 'haarcascade_mcs_rightear.xml'))
nose_cascade = cv2.CascadeClassifier(os.path.join(cascate_root, 'haarcascade_mcs_nose.xml'))

# 'haarcascade_frontalface_alt2.xml'
# 'haarcascade_frontalface_alt_tree.xml'
# 'haarcascade_eye_tree_eyeglasses.xml'

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


# notum queue til að senda svar frá þræði yfir í aðalþráð
q = Queue()

def test_face(mynd, process=True):
    data = utils.image_to_base64png(mynd)
    url = "http://localhost:8080/process/"
    if not process:
        url = "http://localhost:8080/"
    start = time.clock()
    response = requests.post(url, data={'face': base64.b64encode(data)})
    print "response in ", time.clock() - start, " seconds."
    print response.text
    result = json.loads(response.text)
    # print result
    id = result["result"]
    return id


def face_worker(face):
    """ vinnsluþráður gerir ekkert annað en að kalla á fall sem flettir andliti upp á vefþjóni
        þegar svar berst er það sett í queue sem aðalþráður tékkar á í hverri umferð. """
    id = test_face(face)
    q.put(id)


i = 0
id = 0
STEPS = 10

while(True):
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    height, width = frame.shape[:2]
    frame = frame[0:height, width/4:width/4*3]
    framewidth, frameheight = frame.shape[0], frame.shape[1]
    framecopy = None
    found_face = False
    crop_box = None
    i += 1
    if i % 20 == 0:
        framecopy = frame.copy()
    boxcolor = blue        
    if id == 0:
        boxolor = blue
    elif id == -1: 
        boxcolor = red
    else:
        boxcolor = green

    faces = face_cascade2.detectMultiScale(frame, 1.2, 5, 0, (300, 300), (400, 400))
    for (x, y, w, h) in faces:
        found_face = True
        # finna augu
        crop = frame[y:y+h/3*2, x:x+w]
        eyes = eye_cascade1.detectMultiScale(crop, 1.1, 5, 0, (10, 10), (50, 50))
        for x2, y2, w2, h2 in eyes:
             cv2.rectangle(frame, (x+x2, y+y2), (x+x2+w2, y+y2+h2), blue, 1)

        # finna nef
        # nose = nose_cascade.detectMultiScale(frame[y:y+h, x:x+w], 1.3, 5)
        # for x2, y2, w2, h2 in nose:
        #     cv2.rectangle(frame, (x+x2, y+y2), (x+x2+w2, y+y2+h2), blue, 1)
        cropbox = (x-10, min(x+w+10, framewidth-1), y-60, min(y+h+100, frameheight-1))
        cv2.rectangle(frame, (x-10, y-60), (x+w+10, y+h+100), boxcolor, 2)

    frame = cv2.flip(frame, 1)
    if not q.empty():
        id = q.get()
        print "got id {} from thread".format(id)


    cv2.putText(frame, "{} -> {}".format(i, id), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, green)
    cv2.imshow('frame', frame)

    if i % 20 == 0 and found_face and framecopy is not None:
        framecopy = framecopy[cropbox[2]:cropbox[3], cropbox[0]:cropbox[1]]
        thread.start_new_thread(face_worker, (framecopy,))

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('p'):
        # taka mynd og vista hana
        x, y, w, h = faces[0]
        kropp = frame[y-60:y+h+100, x-10:x+w+10]


cap.release()
cv2.destroyAllWindows()
