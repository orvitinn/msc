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
import re
from collections import defaultdict
import base64
import cStringIO

def read_image(image_path):
    return cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

def read_images(path):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list of pairs (id, image)
        id: The corresponding labels (the unique number of the subject, person) in a Python list.
        image: The images, which is a Python list of numpy arrays.
    """
    def process_file(path, filename, res):
        if filename == ".DS_Store":
            return
        try:
            im = cv2.imread(os.path.join(path, filename))
            res.append((id, im))
        except IOError, e:
            print "I/O error: " + str(e)
            raise
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise

    res = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            if not subdirname.startswith("s"):
                continue
            id = int(subdirname[1:])
            subject_path = os.path.join(dirname, subdirname)
            print "reading ", subject_path, " id: ", id
            for filename in os.listdir(subject_path):
                try:
                    process_file(subject_path, filename, res)
                except:
                    pass
    return res


def read_images_from_single_folder(path):
    # lesum inn myndaskrár úr einum folder, nöfn mynda innihalda id og númer.
    # númerið skiptir í sjálfu sér ekki máli en id-ið er nauðsynlegt
    res = []
    for filename in os.listdir(path):
        try:
            if filename.endswith(".png"):
                im = cv2.imread(os.path.join(path, filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)
                id, i, _ = re.split("\D+", filename)
                id, i = int(id), int(i)
                res.append((id, im))
        except ValueError, e:
            print "Error reading filename: ", path, "/", filename
        except IOError, e:
            print "Error opening file: ", filename
    return res


def read_processed_images_from_single_folder(path):
    # lesum inn myndaskrár úr einum folder, nöfn mynda innihalda id og númer.
    # númerið skiptir í sjálfu sér ekki máli en id-ið er nauðsynlegt
    image_list = read_images_from_single_folder(path)
    res_list = []
    for id, img in image_list:
        shape = img.shape
        if len(shape) == 3 and shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            res_list.append((id, img))
    return res_list


def list_processed_images_from_single_folder_with_id(path, id):
    """return a list of full paths of all files beloning to this id"""
    pass


def list_ids_from_single_folder():
    """return a list of all id's in the folder"""
    

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
    return tmp.transform(tmp.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)


def SimpleScaleRotate(image, angle):
    rotated_image = image.rotate(angle)
    return rotated_image


def crop_face(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz=(90,90)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
    offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0*offset_h
    # scale factor
    scale = float(dist)/float(reference)
    if 0.01 < math.fabs(rotation) < 0.3:
        image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    else:
        image = Image.fromarray(image) # þetta var annars gert í ScaleRotate fallinu

    # crop the rotated image
    crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
    crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
    image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image


def center(box):
    # box is a tuple x,y,h,w
    # return a point in the center of the box
    return (box[0] + box[3]/2, box[1] + box[2]/2)


def which_eye(face, eye):
    # face is the bounding box for the face, eye is the bounding box for an eye
    # returns 1 if the eye is the left eye, 2 if it's the right, -1 if we can't tell
    eye_center = center(eye)
    face_center = center(face)
    if eye_center[0] < face_center[0]:
        return 1
    elif eye_center[0] > face_center[0]:
        return 2
    else:
        return -1


class FaceDetector(object):
    def __init__(self):    	# load the cascades
        self.face_cascade = cv2.CascadeClassifier("/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
        self.eyes_cascade = cv2.CascadeClassifier("/opt/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml")
        self.left_eye_cascade = cv2.CascadeClassifier("/opt/local/share/OpenCV/haarcascades/haarcascade_mcs_lefteye.xml")
        self.right_eye_cascade = cv2.CascadeClassifier("/opt/local/share/OpenCV/haarcascades/haarcascade_mcs_righteye.xml")
        # self.eyes_cascade = cv2.CascadeClassifier("/opt/local/share/OpenCV/haarcascades/haarcascade_eye.xml")


    def detectFace(self, img):
        # find a face, return the box for the face and points for left and right eye    # (face, left_eye, right_eye)    
        faces = self.face_cascade.detectMultiScale(img, 1.05, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, minSize=(60,60))
        eyes = None
        eye_res = None
        if len(faces) > 0:
            # kroppum andlit úr mynd
            x, y, h, w = faces[0]
            face_crop = img[y:y+h*2/3, x:x+w]
            eyes = self.eyes_cascade.detectMultiScale(face_crop, 1.2, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (10,10))
            eye_res = []
            if len(eyes) > 0:
                for eye in eyes:
                    ex, ey, eh, ew = eye
                    # hendum út augum sem eru fyrir utan andlit
                    # if ex < x or ex > x+w or ey < y or ey > y+h:
                    #    continue
                    # hendum út ef fyrir neðan miðju
                    # if ey > y+h/2:
                    #    continue
                    # eye_res.append(eye)
                    eye_res.append((x+ex, y+ey, eh, ew))
            if len(eye_res) == 1:
                # fundum eitt auga, en annað vantar. hvort augað fannst? hvorum megin var það í rammanum?
                we = which_eye(faces[0], eye_res[0])
                if we == 1 or we == -1: # left or both
                    # right_face_crop = face_crop[:,center(face_crop)[0]:-1)]
                    right_eye = self.right_eye_cascade.detectMultiScale(face_crop, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (10,10)) # , (80, 80))
                    if len(right_eye):
                        ex, ey, eh, ew = right_eye[0]
                        eye_res.append((x+ex, y+ey, eh, ew))
                elif we == 2 or we == -1: # right or both
                    left_eye = self.left_eye_cascade.detectMultiScale(face_crop, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (10,10)) # , (80,80))
                    if len(left_eye):
                        ex, ey, eh, ew = left_eye[0]
                        eye_res.append((x+ex, y+ey, eh, ew))


        # enn geta verið fleiri augu innan andlits, þarf að henda þeim út

        # print "f:", faces
        # print "e:", eyes
        # return faces, eyes
        return faces, eye_res


def crop_jaw(img):
    """erum með mynd af andliti sem er búið að kroppa og forvinna
       viljum nú teikna yfir hliðar á neðri hluta myndar til að núlla 
       bakgrunn algjörlega út. Sjá bara hvernig kemur út
    """ 
    # punktar_v = np.array([(0, 70), (20, 90), (0, 90)], np.int32)
    # punktar_h = np.array([(90, 70), (70, 90), (90, 90)], np.int32)
    punktar_v = np.array([(0, 45), (5, 80), (20, 90), (0, 90)], np.int32)
    punktar_h = np.array([(90, 45), (85, 80), (70, 90), (90, 90)], np.int32)
    gray = (125, 125, 125)
    cv2.fillConvexPoly(img, punktar_v, gray)
    cv2.fillConvexPoly(img, punktar_h, gray)


def equalize_histogram(img):
    cv2.equalizeHist(img, img)


def check_size_and_resize(img):
    shape = img.shape
    # ef mynd er mjög lítil, stækka hana!
    w, h = shape[0], shape[1]
    min_size = 300
    if w < min_size and h < min_size:
        d = max(float(min_size) / w, float(min_size) / h)
        img = cv2.resize(img, (0, 0), fx=d, fy=d)
    return img


class FaceProcessor(object):
    def __init__(self):
        self.faceDector = FaceDetector()

    def process(self, img):
        return self.process_image(img)

    def process_image(self, img):
        """Processes and crops an image containg a Face

        Args:
            image: an opencv image object containing a Face
        Returns:
            if a face (and eyes) are found
            a new image converted to bw, rotated, cropped and resize
            around the face
            otherwise returns None
        """
        if img is None:
            return None

        shape = img.shape
        if len(shape) == 3 and shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = check_size_and_resize(img)

        faces, eyes = self.faceDector.detectFace(img)
        if faces is None or len(faces) == 0:
            # print "No face found"
            # print ",",
            return None

        # print ".",
        face = faces[0]
        x, y, h, w = face
        if len(eyes) > 1:
            # það virðist ekki alltaf að marka hvort augað er hvað!
            # en vinstra augað er vinstra megin!
            left_eye, right_eye = eyes[0], eyes[1]
            if left_eye[0] > right_eye[0]:
                left_eye, right_eye = right_eye, left_eye

            x, y, w, h = left_eye
            left_eye_center = (x+(w/2), y+(h/2))
            x, y, w, h = right_eye
            right_eye_center = (x+(w/2), y+(h/2))
            cropped_face = crop_face(img, left_eye_center, right_eye_center)
            cropped_face = np.array(cropped_face)
            crop_jaw(cropped_face) # teiknum yfir kjálkalínu fyrir histogram, til að minnka áhrif bakgrunns
            equalize_histogram(cropped_face)
            crop_jaw(cropped_face) # og teiknum aftur yfir kjálkalínuna eftir histogram, til að hún sé alltaf eins
            return cropped_face
        else:
            # print len(eyes), " eyes: ", eyes
            # show_image_and_wait_for_input(img)
            return None


def convert_all_files(input_list):
    """Batch process images

    Args:
        a list containing pairs of (ids, images)
    Returns:
        for each image pair contating a face a new
        image pair with a processed (cropped, rotated, scaled) face
    """
    fp = FaceProcessor()
    res = ((id, fp.process_image(image)) for id, image in input_list)
    return [(id, image) for id, image in res if image is not None]


def convert_single_file(image):
    fp = FaceProcessor()
    image = fp.process_image(image)
    return image


def show_image_and_wait_for_input(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def draw_face(img, face, eyes=None):
    x, y, h, w = face
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for x, y, h, w in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


def save_images(image_list):
    """ fáum inn lista af myndum, vistum myndir """
    # í tmp folder
    path = os.path.join(os.getcwd(), "tmp")
    for i, image in enumerate(image_list):
        filename = "face_{}.png".format(i)
        dest = os.path.join(path, filename)
        cv2.imwrite(dest, image)


def save_images_width_id(image_id_list, destination_folder=None):
    """ fáum inn lista af id, myndum - vistum í folder"""
    # í tmp folder
    if not destination_folder:
        destination_folder = os.path.join(os.getcwd(), "tmp")
    m = defaultdict(int)
    for id, image in image_id_list:
        m[id] += 1
        i = m[id]
        filename = "{}_{}.png".format(id, i)
        # print filename
        destination = os.path.join(destination_folder, filename)
        cv2.imwrite(destination, image)


def image_to_base64png(image):
    # take an (OpenCV) image and return it as a base64 encoded png image
    # http://stackoverflow.com/questions/17682103/how-can-i-send-cv2-frames-to-a-browser
    cnt = cv2.imencode('.png', image)[1]
    return base64.encodestring(cnt)

def base64png_to_image(b64):
    data = base64.decodestring(b64)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)    
    return img

def base64png_to_image_color(b64):
    data = base64.decodestring(b64)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.CV_LOAD_IMAGE_COLOR)    
    return img


def results_to_html(data_file, output_file):
    # opnum skrá með niðurstöðum, löbbum í gegnum gögnin og búum til formaða html skrá sem við skrifum út í aðra skrá
    green = '#40FF00'
    red = '#FE2E2E'
    html_pre = '''<html>\n
    <head><title>Data file - results</title>
        <style>table, th, td {
                  border: 1px solid black;
                  }
        </style>
    </head>\n
    <body>\n<h1>Results</h1>\n'''
    html_post = '</body>\n</html>'
    table_header = '<tr><th></th></tr>\n'
    table_start = '<table>\n'
    table_stop = '</table>\n'
    table_line_big = '<tr bgcolor="{}"><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n'
    table_line_small = '<tr><td>{}</td><td>{}</td></tr>\n'
    description = '<div><p><h2>{}</h2></p><p>Training time: {}, Total prediction time: {} Average prediction time: {}</p></div>\n'

    # same order as in test to format output
    features = ("Fisherfaces", "PCA", "SpatialHistogram", "SpatialHistogram(LPQ())")
    classifiers = (
        "EuclideanDistance",
        "CosineDistance",
        "NormalizedCorrelation",
        "ChiSquareDistance",
        "HistogramIntersection",
        "L1BinRatioDistance",
        "ChiSquareBRD")

    # búa til langan lista sem inniheldur allar samsetningar, í réttri röð
    def next_method_name():
        for f in features:
            for c in classifiers:
                for b in ("TanTriggsPreprocessing", "No preprocessing"):
                    yield "{}, {}, {}".format(f, c, b)

    beginning = True
    next_method = next_method_name()
    with open(data_file, 'r') as data, open(output_file, 'w') as html_file:
        html_file.write(html_pre)
        for row in data.readlines():
            line = row.split(',')
            if len(line) == 2:
                if beginning:
                    beginning = False
                else:
                    html_file.write(table_stop)

                html_file.write(description.format(next_method.next(), line[0], line[1], float(line[1])/11.0))
                html_file.write(table_start)
                # html_file.write(table_line_small.format(line[0], line[1]))
            elif len(line) == 8:
                colour = green
                if int(line[0]) != int(line[1]):
                    colour = red
                html_file.write(table_line_big.format(colour,
                                line[0], line[1], line[2],
                                line[3], line[4], line[5],
                                line[6], line[7]))

        html_file.write(html_post)




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
            crop_face = crop_face(img, left_eye_center, right_eye_center)
            cv2.imshow("Faces found", np.array(crop_face))
            cv2.waitKey(0)
        else:
            cv2.imshow("Faces found", img)
            cv2.waitKey(0)
        # crop the face
        # show
    """
