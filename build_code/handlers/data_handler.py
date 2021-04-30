
#!/usr/bin/env python

"""Functions for downloading and reading MNIST data."""
import pandas as pd
import numpy as np
import random
import re
import json

import os.path
from os import path, listdir
# from os.path import isfile, join

import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataHandler(object):
    def __init__(self, CONFIG):
        self.dataset = None
        self.CONFIG = CONFIG
        # self.prepare_datasets()

    def prepare_datasets(self):

        train_dir = os.path.join(self.CONFIG['DATA_DIR'], self.CONFIG['MODEL_CONFIG']['train_dataset'])
        validation_dir = os.path.join(self.CONFIG['DATA_DIR'],  self.CONFIG['MODEL_CONFIG']['val_dataset']) 

        train_directories = os.listdir(train_dir)
        labels = []
        train_dirs = []
        validation_dirs = []
        total_train = 0
        total_val = 0
        for name in train_directories:
            full_path = os.path.join(train_dir, name)
            # inode = os.stat(full_path)
            if os.path.isdir(full_path):
                labels.append(name)
                train_dirs.append(os.path.join(train_dir, name))
                validation_dirs.append(os.path.join(validation_dir, name))
                total_train = total_train + len(os.listdir(os.path.join(train_dir, name)))
                total_val = total_val + len(os.listdir(os.path.join(validation_dir, name)))

        training_datagen = ImageDataGenerator(rescale=1./255,
                            zoom_range=0.15,
                            horizontal_flip=True,
                            fill_mode='nearest')

        validation_datagen = ImageDataGenerator(rescale = 1./255)

        # data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
        #                             width_shift_range=0.1,
        #                             height_shift_range=0.1,
        #                             #sear_range=0.01,
        #                             zoom_range=[0.9, 1.25],
        #                             horizontal_flip=True,
        #                             vertical_flip=False,
        #                             data_format='channels_last',
        #                             brightness_range=[0.5, 1.5]
        #                         )
                                        
        train_generator = training_datagen.flow_from_directory(
                train_dir,
                target_size=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH']),
                shuffle = True,
                batch_size=77,
                class_mode='categorical')
        
    
        validation_generator = validation_datagen.flow_from_directory(
                validation_dir,
                target_size=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH']),
                batch_size=self.CONFIG['MODEL_CONFIG']['batch_size'],
                shuffle = True,
                class_mode='categorical')       
        print('labels: >> ', labels, ', total_train: >> ', total_train, ', total_val: >> ', total_val)

        # AUTOTUNE = tf.data.AUTOTUNE

        # train_generator = train_generator.prefetch(buffer_size=AUTOTUNE)
        # validation_generator = validation_generator.prefetch(buffer_size=AUTOTUNE)

        self.dataset = DataSet(train_generator, validation_generator, labels, total_train, total_val)  

    def fetch_files(self, DIR_PATH):
        onlyfiles = [f for f in listdir(DIR_PATH) if os.path.isfile(os.path.join(DIR_PATH, f))]
        files = []
        for file in onlyfiles:
            if file == '.DS_Store':
                continue
            file_path = os.path.join(DIR_PATH, file)            
            files.append({'name': file, 'path': file_path})            
        return files    

class DataSet(object):
    def __init__(self, train_generator, validation_generator, labels, total_train, total_val):
        self._train_generator = train_generator
        self._validation_generator = validation_generator
        self._labels = labels
        self._total_train = total_train
        self._total_val = total_val

    @property
    def train_generator(self):
        return self._train_generator

    @property
    def validation_generator(self):
        return self._validation_generator
    
    @property
    def labels(self):
        return self._labels

    @property
    def total_train(self):
        return self._total_train

    @property
    def total_val(self):
        return self._total_val
