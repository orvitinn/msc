#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
# import facerec modules

# import numpy, matplotlib and logging
import numpy as np
from PIL import Image
import logging
import time
import random
import cv2
import math


def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    # c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            if not subdirname.startswith("s"):
                continue
            id = int(subdirname[1:])
            subject_path = os.path.join(dirname, subdirname)
            print "reading ", subject_path, " id: ", id
            i = 0
            for filename in os.listdir(subject_path):
                if filename == ".DS_Store":
                    continue
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(self.sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(id)
                    i += 1
                except IOError, e:
                    print "I/O error: " + str(e)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            print i, " images."
            # c = c+1
    return [X, y]


def Distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx+dy*dy)


def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if scale is None and center is None:
        return image.rotate(angle=angle, resample=resample)

    try:
        nx, ny = x, y = center
    except ValueError, e:
        print "center: " + str(center)
        print e
        raise

    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine/sx
    b = sine/sx
    c = x-nx*a-ny*b
    d = -sine/sy
    e = cosine/sy
    f = y-nx*d-ny*e
    tmp = Image.fromarray(image)
    # return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)
    return tmp.transform(tmp.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)


def SimpleScaleRotate(image, angle):
    # tmp = Image.fromarray(image)
    rotated_image = image.rotate(angle)
    return rotated_image

def CropFace2(image, face, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
    offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0*offset_h
    # scale factor
    scale = float(dist)/float(reference)
    # rotate original around the left eye
    cv2.imshow("ScaleRotateTranslate", np.array(image))
    cv2.waitKey(0)
    print "eye_direction: ", eye_direction, "rotation: ", rotation, ", dist: ", dist, ", reference: ", reference, ", scale: ", scale
    # image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    # image = SimpleScaleRotate(image, rotation)
    # cv2.imshow("ScaleRotateTranslate", np.array(image))
    # cv2.waitKey(0)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
    crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
    print "crop: ", crop_xy, "crop_size: ", crop_size
    # image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
    tmp = Image.fromarray(image)

    image = tmp.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
    # image = tmp.crop((face[0], face[1], face[0]+face[2], face[1]+face[3]))
    image.load()
    # resize it
    # image = SimpleScaleRotate(image, rotation)
    image = image.rotate(math.degrees(-rotation))
    # image = image.resize(dest_sz, Image.ANTIALIAS)
    return image

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
    offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0*offset_h
    # scale factor
    scale = float(dist)/float(reference)
    # rotate original around the left eye
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
    crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
    image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image

class FaceDetector(object):
    def __init__(self):    	# load the cascades
        self.face_cascade = cv2.CascadeClassifier("/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
        self.eyes_cascade = cv2.CascadeClassifier("/opt/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml")

    # find a face, return the box for the face and points for left and right eye    # (face, left_eye, right_eye)    
    def detectFace(self, image):
        faces = self.face_cascade.detectMultiScale(img, 1.1, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
        eyes = None
        if len(faces) > 0:
            eyes = self.eyes_cascade.detectMultiScale(img, 1.1, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
        # print "f:", faces
        # print "e:", eyes
        return faces, eyes



class FaceProcessor(object):
    def __init__(self):
        self.faceDector = FaceDetector()

    def process_image(self, image):
        """Processes and crops an image containg a Face

        Args:
            image: an opencv image object containing a Face
        Returns:
            if a face (and eyes) are found
            a new image converted to bw, rotated, cropped and resize
            around the face
            otherwise returns None
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces, eyes = self.faceDector.detectFace(img)
        for face in faces:
            x, y, h, w = face
            if len(eyes) > 0:
                right_eye, left_eye = eyes[0], eyes[1]
                x, y, w, h = left_eye
                left_eye_center = (x+(w/2), y+(h/2))
                x, y, w, h = right_eye
                right_eye_center = (x+(w/2), y+(h/2))
                crop_face = CropFace(img, left_eye_center, right_eye_center)
                return np.array(crop_face)
            else:
                return None


if __name__ == "__main__":
    # load image and find the face
    # path = '/Users/matti/Documents/forritun/att_faces/'
    path = '/Users/matti/Pictures/'
    # img = cv2.imread(path + "matti.bmp")
    img = cv2.imread(path + "ég2.jpg")
    # convert to bw
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert histogram

    fp = FaceProcessor()
    img = fp.process_image(img)
    cv2.imshow("Faces found", img)
    cv2.waitKey(0)
    """
    fd = FaceDetector()
    faces, eyes = fd.detectFace(img)
    for face in faces:
        x, y, h, w = face
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # for x, y, h, w in eyes:
        #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if len(eyes) > 0:
            right_eye, left_eye = eyes[0], eyes[1]
            print "left: ", left_eye, "right: ", str(right_eye)
            x, y, w, h = left_eye
            left_eye_center = (x+(w/2), y+(h/2))
            x, y, w, h = right_eye
            right_eye_center = (x+(w/2), y+(h/2))
            print "l ", left_eye_center
            print "r ", right_eye_center
            # cv2.line(img, left_eye_center, right_eye_center, (255, 0, 0))
            crop_face = CropFace(img, left_eye_center, right_eye_center)
            cv2.imshow("Faces found", np.array(crop_face))
            cv2.waitKey(0)
        else:
            cv2.imshow("Faces found", img)
            cv2.waitKey(0)
        # crop the face
        # show
    """
