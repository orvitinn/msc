#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import facerec modules
from facerec.feature import Fisherfaces
from facerec.feature import PCA
from facerec.feature import LDA
from facerec.feature import SpatialHistogram

from facerec.preprocessing import TanTriggsPreprocessing
from facerec.operators import ChainOperator
from facerec.model import PredictableModel

from facerec.distance import EuclideanDistance
from facerec.distance import ChiSquareDistance
from facerec.distance import ChiSquareBRD
from facerec.distance import CosineDistance
from facerec.distance import NormalizedCorrelation
from facerec.distance import L1BinRatioDistance
from facerec.distance import HistogramIntersection

from facerec.lbp import LPQ

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

"""
def result_from_res(res):
    # fáum lista af niðurstöðum, viljum meirihluta
    # [3, {'distances': array([ 55.23673802,  55.50875586,  55.8603537 ]), 'labels': array([13, 15,  3])}]
    best_result = res[0]
    distances = res[1]['distances']
    labels = res[1]['labels']
    print res

    unique_labels = set(labels)
    if len(unique_labels) == 1:
        # print "allar niðurstöður eins"
        return best_result

    indexes = np.where(labels == best_result)[0]
    best_distance = min([distances[i] for i in indexes])

    # print res
    # print "best index: ", best_distance
    if len(unique_labels) == 3:
        # þrjár mismunandi niðurstöður, sú best þarf að hafa verið frábær
        if best_distance < 25:
            return best_result
    else:
        if best_distance < 30:
            return best_result

    # fengum ekki góða niðurstöðu
    return -1
"""

def threshold_function(threshold_majority, threshold_unique):
    def result_from_res(res):
        # fáum lista af niðurstöðum, viljum meirihluta
        # [3, {'distances': array([ 55.23673802,  55.50875586,  55.8603537 ]), 'labels': array([13, 15,  3])}]
        best_result = res[0]
        distances = res[1]['distances']
        labels = res[1]['labels']
        print res

        unique_labels = set(labels)
        if len(unique_labels) == 1:
            # print "allar niðurstöður eins"
            return best_result

        indexes = np.where(labels == best_result)[0]
        best_distance = abs(min([distances[i] for i in indexes]))

        # print res
        # print "best index: ", best_distance
        if len(unique_labels) == 3:
            # þrjár mismunandi niðurstöður, sú best þarf að hafa verið frábær
            if best_distance < threshold_unique:
                return best_result
        else:
            if best_distance < threshold_majority:
                return best_result

        # fengum ekki góða niðurstöðu
        return -1

    return result_from_res



if __name__ == "__main__":
    print "starting...."

    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    path = '/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/tmp8'
    if len(sys.argv) > 1:
        path = sys.argv[1]


    # Read in the database, processed face images in a single folder
    start = time.clock()
    input_faces = utils.read_images_from_single_folder(path)
    stop = time.clock()

    print "read ", len(input_faces), ", images from " + path, " in ", stop-start, " seconds."

    prufu_mynd = None

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
          "fisher10": Fisherfaces,
          "pca": PCA,
          "pca10": PCA,
          "lda": LDA,
          "spatial": SpatialHistogram,
          "LPQ" : SpatialHistogram
        }

        if feature_parameter in m: 
            if feature_parameter == 'LPQ':
                feature = SpatialHistogram(LPQ())
            elif feature_parameter == 'fisher80':
                feature = Fisherfaces(80)
            elif feature_parameter == 'pca80':
                feature = PCA(80)
            else:
                feature = m[feature_parameter]()

    # Define a 1-NN classifier with Euclidean Distance:
    # classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=3) # þokkalegt, [1.7472255 ,  1.80661233,  1.89985602] bara fremsta rétt
    # classifier = NearestNeighbor(dist_metric=CosineDistance(), k=3) # þokkalegt, niðurstöður sem mínus tölur ([-0.72430667, -0.65913855, -0.61865271])
    # classifier = NearestNeighbor(dist_metric=NormalizedCorrelation(), k=3) # ágætt  0.28873109,  0.35998333,  0.39835315 (bara fremsta rétt)
    classifier = NearestNeighbor(dist_metric=ChiSquareDistance(), k=3) # gott, 32.49907228,  44.53673458,  45.39480197 bara síðasta rangt
    # classifier = NearestNeighbor(dist_metric=HistogramIntersection(), k=3) # sökkar
    # classifier = NearestNeighbor(dist_metric=L1BinRatioDistance(), k=3) # nokkuð gott,  36.77156378,  47.84164013,  52.63872497] - síðasta rangt
    # classifier = NearestNeighbor(dist_metric=ChiSquareBRD(), k=3) #  36.87781902,  44.06119053,  46.40875114 - síðasta rangt

    # Define the model as the combination
    # model = PredictableModel(feature=feature, classifier=classifier)
    # Compute the model on the given data (in X) and labels (in y):

    feature = ChainOperator(TanTriggsPreprocessing(), feature)
    # classifier = NearestNeighbor()
    model = PredictableModel(feature, classifier)


    # images in one list, id's on another
    id_list, face_list = zip(*input_faces)

    print "Train the model"
    start = time.clock()
    # model.compute(X, y)
    model.compute(face_list, id_list)
    stop = time.clock()
    print "Training done in", stop-start, " next...find a face"

    # test_path = "/Users/matti/Documents/forritun/att_faces/"
    test_path = "/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/test_faces_02"
    """
    target = "10.bmp"
    if len(sys.argv) > 3:
        target = sys.argv[3]
    """

    fp = utils.FaceProcessor()
    
    test_list = [
        ("10.bmp", 41),
        ("matti.png", 41),
        ("matti2.png", 41),
        ("3078.png", 41),
        ("8174.png", 41),
        ("40_1.png", 40),
        ("48_3.png", 48),
        ("51_4.png", 51),
        ("8_7.png", 8),
        ("0455.png", 57),
        ("kolla_01.png", -1),
        ("inga_01.png", -1)
    ]

    """
    test_list = [
        ("10.bmp", 41),
        ("matti.bmp", 41),
        ("3078.jpg", 41),
        ("8174.jpg", 41),
        ("8_7.pgm", 8),
        ("9_6.png", 9),
        ("124_15.png", 124),
        ("12_7.png", 12),
        ("3064.jpg", -1),
        ("inga_01.jpg", -1),
        ("arora_01.png", -1)
    ]
    """

    """
    test_list = [
        ("10.bmp", 41),
        ("matti.bmp", 41),
        ("matti2.jpg", 41),
        ("8_7.pgm", 8),
        ("inga_01.jpg", -1),
        ("kolla_01.jpg", -1),
        ("arora_01.jpg", -1)
    ]
    """

    # threshold_lpq_normalized = threshold_function(0.67, 0.3)
    threshold_lpq_chisquared = threshold_function(70, 35)
    # threshold_spatial_cosine = threshold_function(0.908, 0.908)
    # threshold_spatial_chisuearbrd = threshold_function()

    # threshold = threshold_lpq_normalized
    threshold = threshold_lpq_chisquared
    # threshold = threshold_spatial_cosine

    for image, id in test_list:
        target_full_name = os.path.join(test_path, image)
        prufu_mynd = utils.read_image(target_full_name)
        # prufu_mynd = fp.process_image(utils.read_image(target_full_name))

        if prufu_mynd is not None:
            res = model.predict(prufu_mynd)
            found_id = threshold(res) # result_from_res(res)
            print found_id, ",", id
        else:
            print "Gat ekki opnað prufumynd"

    """
    p1 = fp.process_image(utils.read_image("/Users/matti/Documents/forritun/att_faces/arora_01.jpg"))
    p2 = utils.read_image("/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/test_faces_to_search_for/arora_01.png")
    res1 = model.predict(p1)
    res2 = model.predict(p2)
    print res1
    print res2

    """
    """
    while target != "quit":
        # prufu_mynd = Image.open(os.path.join(path, target))
        target_full_name = os.path.join(test_path, target)
        print "Nota mynd: ", target_full_name
        prufu_mynd = cv2.imread(target_full_name)
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
        else:
            print "gat ekki lesið mynd"
        target = raw_input("Naesta mynd eda quit:")
    """
