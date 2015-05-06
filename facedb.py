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


def threshold_function(threshold_majority, threshold_unique):
    """ fall sem tekur inn stuðla fyrir threshold fall og skilar
        threshold falli sem notar þá stuðla """
    def result_from_res(res):
        """fáum lista af niðurstöðum, viljum meirihluta eða mjög gott gildi
          [3, {'distances': array([ 55.23673802,  55.50875586,  55.8603537 ]),
                  'labels': array([13, 15,  3])}]
         skilum gildi ef það er innan marka, -1 ef ekkert gildi er í lagi """
        suggested_result = res[0]
        distances = res[1]['distances']
        labels = res[1]['labels']

        unique_labels = set(labels)
        if len(unique_labels) == 1:
            return suggested_result

        indexes = np.where(labels == suggested_result)[0]
        best_suggested_distance = abs(min([distances[i] for i in indexes]))

        if len(unique_labels) == 3:
            # þrjár mismunandi niðurstöður, sú best þarf að hafa verið frábær
            if best_suggested_distance < threshold_unique:
                return suggested_result
        else:
            if best_suggested_distance < threshold_majority:
                return suggested_result

        # var besta suggested ekki það sama og besta, var besta betra?
        best_result = min(distances)
        if best_result < best_suggested_distance:
            return labels[np.where(distances == best_result)[0]][0]

        return -1

    return result_from_res


class FaceDatabase(object):

    def __init__(self, database_folder, feature_parameter="LPQ", metric="chi", k=3):
        self.model = None
        
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger("facerec")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        path = database_folder

        start = time.clock()
        input_faces = utils.read_images_from_single_folder(path)
        stop = time.clock()

        print("read {}, images from {} in {} seconds.".format(len(input_faces), path, stop-start))

        feature = None
        m = {
          "fisher": Fisherfaces,
          "fisher80": Fisherfaces,
          "pca": PCA,
          "pca10": PCA,
          "lda": LDA,
          "spatial": SpatialHistogram,
          "LPQ": SpatialHistogram
        }

        if feature_parameter in m:
            if feature_parameter == 'LPQ':
                feature = SpatialHistogram(LPQ())
                self.threshold = threshold_function(71.4, 70)
            elif feature_parameter == 'fisher80':
                feature = Fisherfaces(80)
                self.threshold = threshold_function(0.61, 0.5)
            elif feature_parameter == 'fisher':
                feature = Fisherfaces()
                self.threshold = threshold_function(0.5, 0.3)
            elif feature_parameter == 'pca80':
                feature = PCA(80)
            else:
                feature = m[feature_parameter]()

        metric_param = None
        d = {"euclid": EuclideanDistance,
             "cosine": CosineDistance,
             "normal": NormalizedCorrelation,
             "chi":  ChiSquareDistance,
             "histo": HistogramIntersection,
             "l1b": L1BinRatioDistance,
             "chibrd": ChiSquareBRD
             }
        if metric in d:
            metric_param = d[metric]()
        else:
            metric_param = ChiSquareDistance()

        # Define a 1-NN classifier with Euclidean Distance:
        # classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=3)
        # classifier = NearestNeighbor(dist_metric=CosineDistance(), k=3)
        # classifier = NearestNeighbor(dist_metric=NormalizedCorrelation(),
        # classifier = NearestNeighbor(dist_metric=ChiSquareDistance(), k=3)
        # classifier = NearestNeighbor(dist_metric=HistogramIntersection(), k=3)
        # classifier = NearestNeighbor(dist_metric=L1BinRatioDistance(), k=3)
        # classifier = NearestNeighbor(dist_metric=ChiSquareBRD(), k=3)

        classifier = NearestNeighbor(dist_metric=metric_param, k=k)
        feature = ChainOperator(TanTriggsPreprocessing(), feature)
        self.model = PredictableModel(feature, classifier)

        # images in one list, id's on another
        id_list, face_list = zip(*input_faces)

        print "Train the model"
        start = time.clock()
        # model.compute(X, y)
        self.model.compute(face_list, id_list)
        stop = time.clock()
        print "Training done in", stop-start, " next...find a face"

        # threshold_lpq_normalized = threshold_function(0.67, 0.3)
        # threshold_lpq_chisquared = threshold_function(71.4, 70)
        # threshold_spatial_cosine = threshold_function(0.908, 0.908)
        # threshold_spatial_chisuearbrd = threshold_function()
        # threshold = threshold_lpq_normalized

    def find_face(self, input_face_image):
        assert self.model, "Model is not valid"
        res = self.model.predict(input_face_image)
        print res
        return self.threshold(res)
