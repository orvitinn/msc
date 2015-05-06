#!/usr/bin/env python
# -*- coding: utf-8 -*-


import base64
import threading
import cgi
import requests
import utils
import os

host = "localhost"
port = 8080

def test_face(path, img, process=True):
    mynd = utils.read_image(os.path.join(path, img))
    data = utils.image_to_base64png(mynd)
    url = "http://{}:{}/process/".format(host, port)
    if not process:
        url = "http://{}:{}/".format(host, port)
    response = requests.post(url, data={'face': base64.b64encode(data)})
    print response.text


if __name__ == "__main__":
    test_face('/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/test_faces_02/', '10.png', process=False)
    test_face('/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/sull/', 'simi_t1.jpg')
    test_face('/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/sull/', 'simi_t2.jpg')
    test_face('/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/sull/', 'simi_t3.jpg')
    test_face('/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/sull/', 'simi_t4.jpg')
    test_face('/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/sull/', 'simti_t5_myrkur_skjar.jpg')
