#!/usr/bin/env python

import Augmentor
import numpy as np
import os
import glob
import random
import collections

from PIL import Image

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.datasets import mnist

class ImageUtil(object):
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        print(CONFIG)

    # @staticmethod
    def augmentation(self, path_to_data):
        print("In ImageClassifier, augmentation: {0}".format(path_to_data))
        folders = []
        for f in glob.glob(path_to_data):
            if os.path.isdir(f):
                folders.append(os.path.abspath(f))

        print("Folders (classes) found: %s " % [os.path.split(x)[1] for x in folders])

        pipelines = {}
        for folder in folders:
            print("Folder %s:" % (folder))
            pipelines[os.path.split(folder)[1]] = (Augmentor.Pipeline(folder))
            print("\n----------------------------\n")
        
        for p in pipelines.values():
            print("Class %s has %s samples." % (p.augmentor_images[0].class_label, len(p.augmentor_images)))

        try:
            num_of_samples = int(1000)
            p = Augmentor.Pipeline(path_to_data)
            # Add some operations to an existing pipeline.
            # First, we add a horizontal flip operation to the pipeline:
            p.flip_left_right(probability=0.4)
            # Now we add a vertical flip operation to the pipeline:
            p.flip_top_bottom(probability=0.8)
            # Add a rotate90 operation to the pipeline:
            p.rotate90(probability=0.1)
            p.zoom_random(probability=0.5, percentage_area=0.8)
            # Now we can sample from the pipeline:
            # p.sample(num_of_samples)
            p.flip_top_bottom(probability=0.5)
            p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
            # Now we can sample from the pipeline:
            # p.sample(num_of_samples)

            p2 = Augmentor.Pipeline(path_to_data)
            p2.rotate90(probability=0.5)
            p2.rotate270(probability=0.5)
            p2.flip_left_right(probability=0.8)
            p2.flip_top_bottom(probability=0.3)
            p2.crop_random(probability=1, percentage_area=0.5)
            p2.resize(probability=1.0, width=320, height=320)
            # p2.sample(num_of_samples)
            
        except Exception as e:
            print("Unable to run augmentation: {0}".format(e))

