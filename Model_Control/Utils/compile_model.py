from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing.preprocess import (exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.preprocessing.windowers import create_windows_from_events
import numpy as np
import tensorflow as tf 
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,Callback, CSVLogger, ReduceLROnPlateau
import time
import csv

## OPTIMIZADORES

def getOptimizer(Opt_name:str='adam'):
    """

    Parameters
    ----------
    Opt_name : str, optional
        name of optimizer to use, by default 'adam'
    """

    opt = {
            'Adadelta':tf.keras.optimizers.Adadelta,
            'Adafactor':tf.keras.optimizers.Adafactor,
            'Adagrad':tf.keras.optimizers.Adagrad,
            'Adam':tf.keras.optimizers.Adam,
            'AdamW':tf.keras.optimizers.AdamW,
            'Adamax':tf.keras.optimizers.Adamax,
            'Ftrl':tf.keras.optimizers.Ftrl,
            'Lion':tf.keras.optimizers.Lion,
            'Nadam':tf.keras.optimizers.Nadam,
            'RMSprop':tf.keras.optimizers.RMSprop,
            'SGD':tf.keras.optimizers.SGD,
        }
    
    return opt[Opt_name]


### CALLBACKS

class ThresholdCallback(Callback):
    def __init__(self, threshold):
        super(ThresholdCallback, self).__init__()
        self.threshold = threshold
        #self.log_name  = log_name
    
    def on_epoch_end(self, epoch, logs=None): 
        val_loss = logs['val_loss']
        if val_loss <= self.threshold:
            self.model.stop_training = True

def write_log(filepath='test.log', data=[], mode='w'):
    '''
    filepath: path to save
    data: list of data
    mode: a = update data to file, w = write a new file
    '''
    try:
        with open(filepath, mode) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(data)
    except IOError:
        raise Exception('I/O error')

class TimeHistory(Callback):
    def __init__(self, save_path=None):
        self.save_path = save_path
    def on_train_begin(self, logs={}):
        self.logs = []
        if self.save_path:
            write_log(filepath=self.save_path, data=['time_log'], mode='w')
    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time.time()
    def on_epoch_end(self, epoch, logs={}):
        time_diff = time.time()-self.start_time
        self.logs.append(time_diff)
        if self.save_path:
            write_log(filepath=self.save_path, data=[time_diff], mode='a')

def get_callbacks(callbacks_names,call_args):
    callbacks = dict()
    for i,j in enumerate(callbacks_names):#range(len(callbacks_names)):
        if callbacks_names[j]=='early_stopping':
            callb = EarlyStopping(monitor=call_args[i]['monitor'], patience=call_args[i]['patience'], min_delta=call_args[i]['min_delta'],
                                  mode=call_args[i]['mode'],verbose = call_args[i]['verbose'],restore_best_weights=call_args[i]['restore_best_weights'])
        elif callbacks_names[j]=='checkpoint':
            callb = ModelCheckpoint(filepath=call_args[i]['filepath'],save_format=call_args[i]['save_format'], monitor=call_args[i]['monitor'],
                                    verbose=call_args[i]['verbose'],save_weights_only=call_args[i]['save_weights_only'],save_best_only=call_args[i]['save_best_only'])
        elif callbacks_names[j]=='Threshold':
            callb = ThresholdCallback(threshold=call_args[i]['threshold'])
        elif callbacks_names[j]=='CSVLogger':
            callb = CSVLogger(filename=call_args[i]['csv_dir'])
        elif callbacks_names[j]=='TimeHistory':
            callb = TimeHistory(save_path=call_args[i]['time_log'])
        elif callbacks_names[j]=='ReduceLROnPlateau':
            callb = ReduceLROnPlateau(monitor= call_args[i]['monitor'], patience= call_args[i]['patience'], factor= call_args[i]['factor'], mode= call_args[i]['mode'], verbose=call_args[i]['verbose'], min_lr= call_args[i]['min_lr'])
        callbacks[j]=callb
    return callbacks



def get_loss(name_model:str):
    """

    Parameters
    ----------
    name_model : str
        
    """

    loss_model = {
    
    }

    list_loss = []
    

    return list_loss