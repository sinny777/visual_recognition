#!/usr/bin/env python

#  Author: Gurvinder Singh
#  Date: 2019/08/03
#
# Image Classification.
#
# python build_code/execute.py --data_dir data --result_dir results --config_file model_config.json --data_file data.csv
# 
# *************************************** #

"""A very simple Image classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
from os import environ
import tarfile
import json

import cv2
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# import tensorflow as tf

from utilities.crop_util import CropUtil
# from handlers.keras_model_handler import ModelHandler

FLAGS = None

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def set_config():
    # print(FLAGS)
    if (FLAGS.data_dir[0] == '$'):
      DATA_DIR = os.environ[FLAGS.data_dir[1:]]
    else:
      DATA_DIR = FLAGS.data_dir
    if (FLAGS.result_dir[0] == '$'):
      RESULT_DIR = os.environ[FLAGS.result_dir[1:]]
    else:
      RESULT_DIR = FLAGS.result_dir

    with open(os.path.join(DATA_DIR, FLAGS.config_file), 'r') as f:
        MODEL_CONFIG = json.load(f)

    with open(os.path.join('./secrets', 'secrets.json'), 'r') as f:
        SECRET_CONFIG = json.load(f)
    
    # print(SECRET_CONFIG)

    DATA_FILE_PATH = os.path.join(DATA_DIR, FLAGS.data_file)
    MODEL_PATH = os.path.join(RESULT_DIR, "model", MODEL_CONFIG["model_name"])
    if environ.get('JOB_STATE_DIR') is not None:
        LOG_DIR = os.path.join(os.environ["JOB_STATE_DIR"], MODEL_CONFIG["log_dir"])
    else:
        LOG_DIR = os.path.join(RESULT_DIR, MODEL_CONFIG["log_dir"])
    ensure_dir(DATA_FILE_PATH)
    ensure_dir(MODEL_PATH)
    global CONFIG
    CONFIG = {
                "DATA_DIR": DATA_DIR,
                "RESULT_DIR": RESULT_DIR,
                "DATA_FILE_PATH": DATA_FILE_PATH,
                "MODEL_PATH": MODEL_PATH,
                "LOG_DIR": LOG_DIR,
                "MODEL_CONFIG": MODEL_CONFIG
             }

def execute():
    print('IN execute method >>>>>>>>> ')
    if FLAGS.framework == "scikit":
        print("\n\n <<<<<<<< CREATE MODEL USING SCIKIT LIBRARY >>>>>>>>")
        # model_handler.create_model()        
    elif FLAGS.framework == "tensorflow":
        print("\n\n <<<<<<<< CREATE MODEL USING TENSORFLOW LIBRARY >>>>>>>>")
        # cropUtil = CropUtil(CONFIG)
        cropUtil = CropUtil()
        images = cropUtil.cropImagesAtPath('./data/samples/test/pan/*')
        # images = cropUtil.cropImages(images)
        for image in images:
          # gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
          # image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
          # gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
          # (thresh, image) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
          # (thresh, image) = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
          # cv2.imshow('B&W Image',image)
          # cv2.waitKey(0)
          # cv2.destroyAllWindows()
          plt.figure()
          plt.plot(), plt.imshow(image)
          plt.title('CROP RESULT')
          plt.xticks([]), plt.yticks([])
          plt.show()
        # model_handler.create_model()
    else:
        return None

def main():
    set_config()
    execute()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--data_file', type=str, default='data.csv', help='Data file name')
  parser.add_argument('--config_file', type=str, default='model_config.json', help='Model Configuration file name')
  parser.add_argument('--framework', type=str, default='tensorflow', help='ML Framework to use')

  FLAGS, unparsed = parser.parse_known_args()
  print("Start model training")
#   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  main()
