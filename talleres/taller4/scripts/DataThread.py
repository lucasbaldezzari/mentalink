# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 18:25:00 2021

@author: enfil
"""

"""
Created on Wed Jun 23 11:27:24 2021

@author: Lucas BALDEZZARI

"""
import sys
import os

import warnings
import numpy as np
import numpy.matlib as npm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import KFold

import tensorflow as tf

from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from tensorflow.keras import initializers, regularizers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy

from brainflow.data_filter import DataFilter

from utils import filterEEG, segmentingEEG, computeMagnitudSpectrum, computeComplexSpectrum
from utils import plotSpectrum, plotOneSpectrum
from CNNClassify import CNNClassify
import time
import brainflow
import numpy as np
import threading

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

class DataThread(threading.Thread):
    
    def __init__(self, board, board_id):
        threading.Thread.__init__(self)
        
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.keep_alive = True
        self.board = board
        
    def run (self):
        
        window_size = 4 #secs
        sleep_time = 1 #secs
        points_per_update = window_size * self.sampling_rate
        
        while self.keep_alive:
            time.sleep (sleep_time)
            # get current board data doesnt remove data from the buffer
            data = self.board.get_current_board_data(int(points_per_update))
<<<<<<< Updated upstream
            # print(f"New data {data[:10]}")
            print(f"New data {data.shape}")
=======
            #print(f"New data {data[:10]}")
            
            
            import fileAdmin as fa
            
            actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
            path = os.path.join(actualFolder,"dataset")
            
            subjects = [8]
                
            fm = 256.0
            tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
            muestraDescarte = 39
            frecStimulus = np.array([9.25, 11.25, 13.25,
                                     9.75, 11.75, 13.75,
                                     10.25, 12.25, 14.25,
                                     10.75, 12.75, 14.75])

            
            data = data[:1063]
            rawEEG = data
            #get the selected trial and stimulus from rawEEG
            actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
            path = os.path.join(actualFolder,"models")
            samples = 1024
            #samples = rawEEG.shape[2]
            resolution = 256/1024
            
            rawEEG = rawEEG[:,:, muestraDescarte: ,:]
            rawEEG = rawEEG[:,:, :tiempoTotal ,:]
            
            PRE_PROCES_PARAMS = {
                            'lfrec': 3.,
                            'hfrec': 36.,
                            'order': 4,
                            'sampling_rate': fm,
                            'window': 4,
                            'shiftLen':4
                            }
            
            
            FFT_PARAMS = {
                            'resolution': resolution,#0.2930,
                            'start_frequency': 5.0,
                            'end_frequency': 38.0,
                            'sampling_rate': fm
                            }
            
            """
            **********************************************************************
            First step: Create CNNClassify object in order to load a trained model
            and classify new data
            **********************************************************************
            """
            
            # create an CNNClassify object in order to work with magnitud features
            magnitudcomplexCNNClassifier = CNNClassify(modelFile = "CNN_UsingMagnitudFeatures_Subject8",
                                       weightFile = "bestWeightss_CNN_UsingMagnitudFeatures_Subject8",
                                       PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                       FFT_PARAMS = FFT_PARAMS,
                                       classiName = f"CNN_Classifier",
                                       frecStimulus = frecStimulus.tolist())
            
            # create an CNNClassify object in order to work with magnitud and complex features
            complexCNNClassifier = CNNClassify(modelFile = "CNN_UsingComplexFeatures_Subject8",
                                       weightFile = "bestWeightss_CNN_UsingComplexFeatures_Subject8",
                                       PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                                       FFT_PARAMS = FFT_PARAMS,
                                       classiName = f"CNN_Classifier",
                                       frecStimulus = frecStimulus.tolist())
            
            """
            **********************************************************************
            Second step: Create CNNClassify object in order to load a trained model
            and classify new data
            **********************************************************************
            """
            
            # get the features for my data
            magnitudFeatures = magnitudcomplexCNNClassifier.computeMSF(data)
            complexFeatures = complexCNNClassifier.computeCSF(data)
            
            # Prepare my data for classification. This is important, the input data for classification
            # must be the same shape the CNN was trained.
            complexDataForClassification = complexCNNClassifier.getDataForClassification(complexFeatures)
            magnitudDataForClassification = magnitudcomplexCNNClassifier.getDataForClassification(magnitudFeatures)
            
            # Get a classification. The classifyEEGSignal() method give us a stimulus
            complexClassification = complexCNNClassifier.classifyEEGSignal(complexDataForClassification)
            magnitudClassification = magnitudcomplexCNNClassifier.classifyEEGSignal(magnitudDataForClassification)
            
            
            print("The stimulus classified using magnitud features is: ", magnitudClassification)
            print("The stimulus classified using complex features is: ", complexClassification)
>>>>>>> Stashed changes
            
def main():
    BoardShim.enable_dev_board_logger()
    
    # use synthetic board for demo only
    params = BrainFlowInputParams ()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = BoardShim (board_id, params)
    board.prepare_session ()
    board.start_stream ()
    
    data_thread = DataThread(board, board_id) #Creo un  objeto del tipo DataThread
    data_thread.start () #Se ejecuta el método run() del objeto data_thread
    
    try:
        time.sleep(4)
        
    finally:
        data_thread.keep_alive = False
        data_thread.join() #free thread
        
    board.stop_stream()
    board.release_session()
    
if __name__ == "__main__":
    main()
    
    
    
    
    
            