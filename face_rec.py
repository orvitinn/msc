#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import facerec modules
from facerec.feature import Fisherfaces
from facerec.feature import PCA
from facerec.feature import LDA
from facerec.feature import SpatialHistogram

from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
# from facerec.visual import subplot
from facerec.util import minmax_normalize

# import numpy, matplotlib and logging
import numpy as np
from PIL import Image
# import matplotlib.cm as cm
import cv2

import logging
import time
import random
import sys
import os

import utils


'''
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
'''

if __name__ == "__main__":
    print "starting...."

    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    path = '/Users/matti/Documents/forritun/att_faces/'
    if len(sys.argv) > 1:
        path = sys.argv[1]

    print "reading images from " + path
    # Now read in the image data. This must be a valid path!
    # [X, y] = read_images(path)
    input_faces = utils.read_images(path)

    # use a random image
    # random.seed()
    # r = len(X)
    # random_index = random.randint(0, r-1)

    # print "using image ", random_index, " id: ", y[random_index]

    prufu_mynd = None

    # test data and label
    # prufu_mynd, tl = X[random_index], y[random_index]
    # and remove test from the data
    # del X[random_index]
    # del y[random_index]

    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Define the Fisherfaces as Feature Extraction method:

    # default
    feature = Fisherfaces()

    if len(sys.argv) > 2:
        feature_parameter = sys.argv[2]
        m = {
          "fisher": Fisherfaces,
          "pca": PCA,
          "lda": LDA,
          "spatial": SpatialHistogram
        }

        if feature_parameter in m:
            feature = m[feature_parameter]()

    # Define a 1-NN classifier with Euclidean Distance:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    # Define the model as the combination
    model = PredictableModel(feature=feature, classifier=classifier)
    # Compute the model on the given data (in X) and labels (in y):

    print "processing input images, ", len(input_faces)
    input_faces = utils.convert_all_files(input_faces)
    #  remove null faces
    # input_faces = [(a, b) for a, b in input_faces if b is not None]
    print "nulls removed, ", len(input_faces)

    # images in one list, id's on another
    id_list, face_list = zip(*input_faces)

    # print "saving images"
    # utils.save_images(face_list)

    # show random image
    # utils.show_image_and_wait_for_input(face_list[len(face_list)/2])

    print "Train the model"
    start = time.clock()
    # model.compute(X, y)
    model.compute(face_list, id_list)
    stop = time.clock()
    print "Training done in", stop-start, " next...find a face"

    target = "10.bmp"
    if len(sys.argv) > 3:
        target = sys.argv[3]

    fp = utils.FaceProcessor()

    while target != "quit":
        # prufu_mynd = Image.open(os.path.join(path, target))
        prufu_mynd = cv2.imread(os.path.join(path, target))
        print "Nota mynd: ", os.path.join(path, target)
        if prufu_mynd is not None:
            prufu_mynd = fp.process_image(prufu_mynd)
            if prufu_mynd is None:
                print "fann ekkert andlit!"
            else:
                start = time.clock()
                # res = model.predict(td)
                res = model.predict(prufu_mynd)
                stop = time.clock()
                print res
                print "time: ", stop-start
        target = raw_input("Naesta mynd eda quit:")
