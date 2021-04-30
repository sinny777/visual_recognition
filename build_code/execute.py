#!/usr/bin/env python

#  Author: Gurvinder Singh
#  Date: 14/04/2021
#
# Image Classification.
#
# export DATA_DIR=~/Documents/Development/data/ml/datasets/cctv_detection/dataset2/test
# python build_code/execute.py --result_dir results --config_file config/config.json
# python build_code/execute.py --data_dir data/cctv_detection/dataset1 --result_dir results --config_file config/config.json
#
# *************************************** #

"""A very simple Image classifier using Tensorflow 2.
See extensive documentation at
https://www.tensorflow.org/tutorials/images/classification
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt

import argparse
import sys
import os
from os import environ
import tarfile
import json

from handlers.model_handler import ModelHandler

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

    # print(MODEL_CONFIG)

    # DATA_FILE_PATH = os.path.join(DATA_DIR, FLAGS.data_file)
    MODEL_PATH = os.path.join(RESULT_DIR, "model", MODEL_CONFIG["model_name"])
    CHECKPOINTS_PATH = os.path.join(RESULT_DIR, "checkpoints/", MODEL_CONFIG["checkpoints_dir"])
    if environ.get('JOB_STATE_DIR') is not None:
        LOG_DIR = os.path.join(os.environ["JOB_STATE_DIR"], MODEL_CONFIG["log_dir"])
    else:
        LOG_DIR = os.path.join(RESULT_DIR, MODEL_CONFIG["log_dir"])
    # ensure_dir(DATA_FILE_PATH)
    ensure_dir(MODEL_PATH)
    global CONFIG
    CONFIG = {
                "DATA_DIR": DATA_DIR,
                "RESULT_DIR": RESULT_DIR,
                "MODEL_PATH": MODEL_PATH,
                "LOG_DIR": LOG_DIR,
                "MODEL_CONFIG": MODEL_CONFIG,
                "CHECKPOINTS_PATH": CHECKPOINTS_PATH
             }
    
def main():
    set_config()
    model_handler = ModelHandler(CONFIG) 
    if FLAGS.action == 'train':
        model_handler.clean()
        model_handler.data_handler.prepare_datasets()
        model = model_handler.create_model(CONFIG['MODEL_CONFIG']['base_model_name'])
        if model != None:
           trained_model, history1 = model_handler.train_model(model, CONFIG['MODEL_CONFIG']['epochs'])
           model_handler.save_model(trained_model)
           retrained_model, history2 = model_handler.retrain_model(trained_model, 10)
           model_handler.save_model(retrained_model)
           model_handler.plot_history(history2, CONFIG['MODEL_CONFIG']['epochs'] ) 
          #  label_dict_l = model_handler.get_label_dict(train_generator) 
    elif FLAGS.action == 'retrain':
        model_handler.clean()
        model_handler.data_handler.prepare_datasets()
        model = model_handler.load_model()
        retrained_model, history = model_handler.retrain_model(model, 10)
        model_handler.save_model(retrained_model)
        model_handler.plot_history(history, 10) 
    elif FLAGS.action == 'convert':
        model_handler.save_tflite_model()
    elif FLAGS.action == 'classify':
        test_dir = os.path.join(CONFIG['DATA_DIR'], 'test/fire')
        images = model_handler.data_handler.fetch_files(test_dir)
        results1 = model_handler.predict_image(images)
        for result in results1:
          print(result)
        print('\n\n')   
        # results2 = model_handler.predict_tflite(images)             
        # for result in results2:
        #   print(result)
        # print('\n\n')
        # image_path = os.path.join(CONFIG['DATA_DIR'], 'cctv_detection/test/cctv_fire1.jpeg')
    else:
      model_handler.data_handler.prepare_datasets()
      print(model_handler.data_handler.dataset._labels)
      print('NO ACTIONS TO PERFORM, PLZ provide an action to do !! ')           

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--config_file', type=str, default='config/config.json', help='Model Configuration file name')
  parser.add_argument('--action', type=str, default='classify', help='Action to perform')

  FLAGS, unparsed = parser.parse_known_args()
#   print("Start model training")
#   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  main()  
