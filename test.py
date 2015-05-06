#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from PIL import Image
import cv2

import logging
import time
import random
import sys
import os

import utils
from facedb import FaceDatabase

def test_processing():
    # load images
    image_list = utils.read_images('/Users/matti/Documents/forritun/andlit2/')
    print len(image_list)
    # pre_ids, pre_images = zip(*image_list)
    processed_images = utils.convert_all_files(image_list)
    print len(processed_images)
    # ids, images = zip(*processed_images)
    # print " pre: ", pre_ids
    # print "post: ", ids
    # utils.save_images(images)
    utils.save_images_width_id(processed_images)

def test_detection():
    image_list = utils.read_images('/Users/matti/Documents/forritun/andlit2/s18')
    fd = utils.FaceDetector()
    ids, images = zip(*processed_images)



def test_one_face(target):
    # prófum þetta, finnst andlit í einfaldri mynd?
    # path = '/Users/matti/Documents/forritun/andlit2/s41'
    # path = '/Volumes/MacSarpur/Downloads/colorferet/colorferet/dvd1/data/images/s/s00001'
    path = '/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/sull'
    #path = '/Users/matti/Pictures/'
    # img = cv2.imread(path + "matti.bmp")
    # target = "ég2.jpg"
    image = cv2.imread(os.path.join(path, target))
    if image is None:
        print "Gat ekki lesið mynd"
        return
    # print image.shape
    image = utils.check_size_and_resize(image)

    df = utils.FaceDetector()
    face, eyes = df.detectFace(image)
    if face is None or len(face) == 0:
        print "Fann ekkert andlit í mynd: ", target
        utils.show_image_and_wait_for_input(image)
        return

    face = face[0]
    print "f:",face, "e:", eyes
    image2 = image.copy()
    utils.draw_face(image2, face, eyes)
    utils.show_image_and_wait_for_input(image2)
    fp = utils.FaceProcessor()
    image2 = fp.process_image(image)
    assert image2 is not None, "Mynd á ekki að vera null"
    utils.show_image_and_wait_for_input(image2)
    print "done"

def test_failed_face():
    test_one_face('6.bmp' )


def test_reading_from_single_folder():
    import random 
    images = utils.read_images_from_single_folder("/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/tmp")
    if images:
        id, image = random.choice(images)
        print id
        utils.show_image_and_wait_for_input(image)
    else:
        print "no images"

def print_dimension(image):
    shape = image.shape
    print shape

def check_processed_image():
    images = utils.read_images_from_single_folder("/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/tmp")
    if images:
        id, image = random.choice(images)
        # print id
        # utils.show_image_and_wait_for_input(image)
        print "processed: ",
        print_dimension(image)

def check_unprocessed_image():
    image = utils.read_image("/Users/matti/Documents/forritun/att_faces/10.bmp")
    fp = utils.FaceProcessor()
    image2 = fp.process_image(image)
    print "unprocessed: ",
    print_dimension(image2)

def check_histogram():
    image = utils.read_image("/Users/matti/Documents/forritun/att_faces/matti2.jpg")
    utils.show_image_and_wait_for_input(image)
    fp = utils.FaceProcessor()
    image2 = fp.process_image(image)
    utils.show_image_and_wait_for_input(image2)
    utils.equalize_histogram(image2)
    utils.show_image_and_wait_for_input(image2)


def test_draw():
    img  = utils.read_image("/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/test_faces_to_search_for/arora_01.jpg")
    utils.crop_jaw(img)
    utils.show_image_and_wait_for_input(img)

def test_compare_images(img1_filename, img2_filename):
    fp = utils.FaceProcessor()
    img1 = fp.process_image(utils.read_image(img1_filename))
    img2 = utils.read_image(img2_filename)

    print img1
    print "---"
    print img2

    print img1.shape, ", ", img2.shape

    img3 = img1 - img2
    print img3.sum() # allt núll, nákvæmlega eins
    img3 = img2 - img1
    print img3.sum() # allt núll, nákvæmlega eins

    # utils.show_image_and_wait_for_input(img3)

def test_tanprocessing(filename):
    from facerec.preprocessing import TanTriggsPreprocessing
    path = '/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/test_faces_02'
    image = cv2.imread(os.path.join(path, filename))
    utils.show_image_and_wait_for_input(image)
    t = TanTriggsPreprocessing()
    res = t.extract(image)
    utils.show_image_and_wait_for_input(res)

def test_face_database():
    fdb = FaceDatabase("/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/tmp8", "LPQ")
    img = utils.read_image("/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/test_faces_02/3078.png")
    print "find_face returned: ", fdb.find_face(img)


def test_base64png():
    img = utils.read_image("/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/test_faces_02/3078.png")
    b64 = utils.image_to_base64png(img)
    print b64[:100]
    img2 = utils.base64png_to_image_color(b64)
    utils.show_image_and_wait_for_input(img2)


if __name__ == "__main__":
    # check_histogram()
    # check_processed_image()
    #  check_unprocessed_image()
    # test_processing()
    # test_failed_face()
    # test_reading_from_single_folder()

    # test_draw()
    # test_tanprocessing('0455 .png')
    # test_face_database()
    # test_base64png()

    test_one_face('simi_t1.jpg')
    test_one_face('simi_t2.jpg')
    test_one_face('simi_t3.jpg')
    test_one_face('simi_t4.jpg')
    # test_one_face('gyda_03.jpg')
    # test_one_face('gyda_05.jpg')
    #test_compare_images(
    #    '/Users/matti/Documents/forritun/att_faces/arora_01.jpg', 
    #    '/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/test_faces_to_search_for/arora_01.png'
    #)

    """
    test_one_face('3.bmp')
    test_one_face('4.bmp')
    test_one_face('5.jpg')
    test_one_face('6.bmp')
    test_one_face('7.bmp')
    test_one_face('8.bmp')
    test_one_face('9.bmp')
    """
