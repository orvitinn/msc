#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import utils
import sys
import cv2

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

def process_single_image(input_file, destination_path):
    image = utils.read_image(input_file)
    image = utils.convert_single_file(image)
    if image is None:
        print "Error converting image"
        sys.exit(-1)
    filename = input_file[input_file.rfind("/")+1:] # hvað gerist ef það er engin slóð í path?
    filename = filename[:-4] + ".png"
    destination = os.path.join(destination_path, filename)
    CV_IMWRITE_PNG_COMPRESSION = 16
    cv2.imwrite(destination, image, (CV_IMWRITE_PNG_COMPRESSION, 0))

def process(input_path, destination_path, id_delta=0):
    print "processing ", input_path
    images = utils.read_images(input_path)
    print len(images), " images read"
    new_images = utils.convert_all_files(images)
    updated_list = [(id+id_delta, image) for id, image in new_images]
    print "saving"
    utils.save_images_width_id(updated_list, destination_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print """Usage: process input output
        where input is the root
        of the face database folder
        and output is the destination
        folder for allt the files"""
        sys.exit(-1)

    if len(sys.argv) == 3:
        process(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[3] == 'single':
        process_single_image(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[3] == 'html':
        utils.results_to_html(sys.argv[1], sys.argv[2])
