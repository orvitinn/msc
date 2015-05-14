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
from facerec.util import minmax_normalize

import numpy as np
from PIL import Image
import cv2

import logging
import time
import random
import sys
import os

import utils

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

def test_one_method(input_faces, test_faces, feature, classifier, chain=True):
    if chain:
        feature = ChainOperator(TanTriggsPreprocessing(), feature)

    model = PredictableModel(feature, classifier)
    id_list, face_list = zip(*input_faces)

    start = time.clock()
    model.compute(face_list, id_list)
    stop = time.clock()
    training_time = stop-start

    res_list = []
    start = time.clock()
    for id, image in test_faces:
        res = model.predict(image)
        res_list.append([id]+res)
    stop = time.clock()
    predict_time = stop-start

    return (training_time, predict_time, res_list)


if __name__ == "__main__":
    print "starting...."

    path = '/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/tmp8'
    if len(sys.argv) > 1:
        path = sys.argv[1]

    # Read in the database, processed face images in a single folder
    input_faces = utils.read_images_from_single_folder(path)
    print "read ", len(input_faces), ", images from " + path

    '''
    test_list = (
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
    )'''

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

    test_path = "/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/test_faces_02"
    # test_path = "/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/test_faces_to_search_for"
    test_faces = []
    for image, id in test_list:
        test_faces.append((id, utils.read_image(os.path.join(test_path, image))))

    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # ætla að prófa allar aðferðir
    # feature = Fisherfaces()
    m = (Fisherfaces(), PCA(), SpatialHistogram(), SpatialHistogram(LPQ()))

    classifiers = (
        # Define a 1-NN classifier with Euclidean Distance:
        NearestNeighbor(dist_metric=EuclideanDistance(), k=3),
        NearestNeighbor(dist_metric=CosineDistance(), k=3),
        NearestNeighbor(dist_metric=NormalizedCorrelation(), k=3),
        NearestNeighbor(dist_metric=ChiSquareDistance(), k=3),
        NearestNeighbor(dist_metric=HistogramIntersection(), k=3),
        NearestNeighbor(dist_metric=L1BinRatioDistance(), k=3),
        NearestNeighbor(dist_metric=ChiSquareBRD(), k=3),
    )

    def test_one(idx):
        tt, pt, res_list = test_one_method(input_faces, test_faces, m[idx], classifiers[idx], True)
        print tt, ",", pt
        for id, guess, rm in res_list:
            labels = rm['labels']
            distances = rm['distances']
            # print id, guess, labels[0], labels[1], labels[2], distances[0], distances[1], distances[2]
            print('{},{},{},{},{},{},{},{},'.format(id, guess, labels[0], labels[1], labels[2], distances[0], distances[1], distances[2]))

    # test_one(0)
    # test_one(1)
    print "starting test"
        with open('data.txt', 'w', 0) as data_file:
        for feature in m:
            for classifier in classifiers:
                for chain in (True, False):
                    print ':',
                    tt, pt, res_list = test_one_method(input_faces, test_faces, feature, classifier, chain)
                    data_file.write('{}, {}\n'.format(tt, pt))
                    print tt, pt
                    for real_id, guess, rm in res_list:
                        labels = rm['labels']
                        distances = rm['distances']
                        data_file.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(real_id, guess, labels[0], labels[1], labels[2], distances[0], distances[1], distances[2]))

