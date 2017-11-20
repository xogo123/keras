import sys
import os
import time
import argparse

import json
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *
from keras.utils import plot_model
from keras import backend as K

try :
    import matplotlib
    import matplotlib.pyplot as plt
except :
    print ('no pyplot')

#dict_arg['model_name'] = get_model_name(dict_arg['model_type'])

def get_path_new_model(model_type):
    k = 0
    while os.path.isfile('./model/{}_{}.h5'.format(model_type,k)) :
        k += 1
    return './model/{}_{}.h5'.format(model_type,k)

def get_args():
    parser = argparse.ArgumentParser(description='run model')
    
    parser.add_argument('dir_data', type=str, help='path of data')
    parser.add_argument('path_output_1', type=str, help='path of output_1')
    parser.add_argument('path_output_2', type=str, help='path of output_2')
    parser.add_argument('--dir_model', type=str, help='path of model')
    
    parser.set_defaults(dir_data='./data/')
    parser.set_defaults(path_output_1='./output_1.txt')
    parser.set_defaults(path_output_2='./output_2.txt')
    parser.set_defaults(dir_model='./model/')
    
    args = parser.parse_args() 
    
    return args

def tensorflow_init() :
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)
    
def lst_keras_callbacks(path_new_model) :
    MCP = keras.callbacks.ModelCheckpoint(path_new_model, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    ES = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=3, verbose=0, mode='auto')
    
    return [MCP,ES]
