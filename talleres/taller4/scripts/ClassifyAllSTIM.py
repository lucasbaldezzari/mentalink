# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 13:52:44 2021

@author: enfil
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

class CNNClassify():
    
    def __init__(self, modelFile = "CNNModel", weightFile = "weightFile",
                 PRE_PROCES_PARAMS = dict(), FFT_PARAMS = dict(), classiName = "",
                 frecStimulus = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]):
        """
        Some important variables configuration and initialization in order to implement a CNN 
        model for CLASSIFICATION.
        
        Args:
            - modelFile: File's name to load the pre trained model.
            - weightFile: File's name to load the pre model's weights
            - PRE_PROCES_PARAMS: The params used in order to pre process the raw EEG.
            - FFT_PARAMS: The params used in order to compute the FFT
            - CNN_PARAMS: The params used for the CNN model.
        """
        
        # load model from JSON file
        with open(f"models/{modelFile}.json", "r") as json_file:
            loadedModel = json_file.read()
        
            self.loadedModel = model_from_json(loadedModel)
            
        self.loadedModel.load_weights(f"models/{weightFile}.h5")
        self.loadedModel.make_predict_function()
        
        self.CSF = np.array([]) #Momplex Spectrum Features
        self.MSF = np.array([]) #Magnitud Spectrum Features
        
        self.classiName = classiName #Classfier object name
        
        self.frecStimulusList = frecStimulus
        
    #Setting variables for EEG processing.
        if not PRE_PROCES_PARAMS:
            self.PRE_PROCES_PARAMS = {
                'lfrec': 3.,
                'hfrec': 80.,
                'order': 4,
                'sampling_rate': 256.,
                'window': 4,
                'shiftLen':4
                }
        else:
            self.PRE_PROCES_PARAMS = PRE_PROCES_PARAMS
             
        #Setting variables for the Magnitude and Complex features extracted from FFT
        if not FFT_PARAMS:        
            self.FFT_PARAMS = {
                'resolution': 0.2930,
                'start_frequency': 0.0,
                'end_frequency': 38.0,
                'sampling_rate': 256.
                }
        else:
            self.FFT_PARAMS = FFT_PARAMS

        
    def classifyEEGSignal(self, dataForClassification):
        """
        Method used to classify new data.
        
        Args:
            - dataForClassification: Data for classification. The shape must be
            []
        """
        print(dataForClassification.shape)
        self.preds = self.loadedModel.predict(dataForClassification)
        
        return self.frecStimulusList[np.argmax(self.preds[0])]
    
    def computeMSF(self, rawEEG):
            """
            Compute the FFT over segmented EEG data.
            
            Argument:
                - None. This method use variables from the own class
            
            Return:
                - The Magnitud Spectrum Feature (MSF).
            """
            
            #eeg data filtering
            filteredEEG = filterEEG(rawEEG, self.PRE_PROCES_PARAMS["lfrec"],
                                    self.PRE_PROCES_PARAMS["hfrec"],
                                    self.PRE_PROCES_PARAMS["order"],
                                    self.PRE_PROCES_PARAMS["sampling_rate"])
            
            #eeg data segmentation
            eegSegmented = segmentingEEG(filteredEEG, self.PRE_PROCES_PARAMS["window"],
                                         self.PRE_PROCES_PARAMS["shiftLen"],
                                         self.PRE_PROCES_PARAMS["sampling_rate"])
            
            self.MSF = computeMagnitudSpectrum(eegSegmented, self.FFT_PARAMS)
            
            return self.MSF
        
    def computeCSF(self, rawEEG):
            """
            Compute the FFT over segmented EEG data.
            
            Argument:
                - None. This method use variables from the own class
            
            Return:
                - The Complex Spectrum Feature (CSF) and the MSF in the same matrix
            """
            
            #eeg data filtering
            filteredEEG = filterEEG(rawEEG, self.PRE_PROCES_PARAMS["lfrec"],
                                    self.PRE_PROCES_PARAMS["hfrec"],
                                    self.PRE_PROCES_PARAMS["order"],
                                    self.PRE_PROCES_PARAMS["sampling_rate"])
            
            #eeg data segmentation
            #shape: [targets, canales, trials, segments, duration]
            eegSegmented = segmentingEEG(filteredEEG, self.PRE_PROCES_PARAMS["window"],
                                         self.PRE_PROCES_PARAMS["shiftLen"],
                                         self.PRE_PROCES_PARAMS["sampling_rate"])
            # print("Segmented EEG: ", eegSegmented.shape)
            
            #shape [2*n_fc, num_channels, num_classes, num_trials, number_of_segments]
            self.CSF = computeComplexSpectrum(eegSegmented, self.FFT_PARAMS)
            # print("self.CSF shape: ", self.CSF.shape)
            
            return self.CSF

    def getDataForClassification(self, features):
        """
        Prepare the features set in order to fit the CNN model and get a classification.
        
        Arguments:
            - features: Magnitud Spectrum Features or Complex Spectrum Features with shape
            
            [targets, channels, trials, segments, samples].
        
        """
        
        print("Generating data for classification")
        print("Original features shape: ", features.shape)
        featuresData = np.reshape(features, (features.shape[0], features.shape[1],features.shape[2],
                                             features.shape[3]*features.shape[4]))
        
        # print("featuresData shape: ", featuresData.shape)
        
        dataForClassification = featuresData[:, :, 0, :].T
        # print("Transpose trainData shape(1), ", dataForClassification.shape)
        
        #Reshaping the data into dim [classes*trials x channels x features]
        for target in range(1, featuresData.shape[2]):
            dataForClassification = np.vstack([dataForClassification, np.squeeze(featuresData[:, :, target, :]).T])
            
        # print("trainData shape (2), ",dataForClassification.shape)
    
        dataForClassification = np.reshape(dataForClassification, (dataForClassification.shape[0], dataForClassification.shape[1], 
                                             dataForClassification.shape[2], 1))
        
        # print("Final trainData shape (3), ",dataForClassification.shape)
        
        return dataForClassification
        
def main():
    
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"dataset")
    # dataSet = sciio.loadmat(f"{path}/s{subject}.mat")
    
# if __name__ == "__main__":
#     main()

import fileAdmin as fa

actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
path = os.path.join(actualFolder,"dataset")

subjects = list(range(1,11))
    
fm = 256.0
tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
muestraDescarte = 39
frecStimulus = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75])


"""
**********************************************************************
Loading and plotting the EEG

IMPORTANT: In real time BCI, the "loading data" will be replaced
for real data coming from the OpenBCI board
**********************************************************************
"""

datos = {}

for s in subjects:
    for i in list(range(12)):
        magnitud = []
        complejo = []
        for t in list(range(3)):
            rawEEG = fa.loadData(path = path, subjects = subjects)[f"s{s}"]["eeg"]
            
            #selec the last 3 trials
            rawEEG = rawEEG[:, :, :, 12:]
            
            stimulus = i #slected stimulus for classification
            trial = t #selected trial
             
            #get the selected trial and stimulus from rawEEG
            data = rawEEG[stimulus,:,:,trial].reshape(1, rawEEG.shape[1],rawEEG.shape[2],1)
            path = os.path.join(actualFolder,"dataset")
            
            samples = rawEEG.shape[2]
            resolution = fm/samples
            
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
            
            '''
            plotOneSpectrum(magnitudFeatures, resolution, 12, s, 5, [magnitudClassification],
                          startFrecGraph = FFT_PARAMS['start_frequency'],
                          save = False,
                          title = f"Stimulus classified using magnitud features: {magnitudClassification}", folder = "figs")
            '''
            
            if magnitudClassification == frecStimulus[i]:
                magnitud.append(1) 
            else:
                magnitud.append(0)
            
            if complexClassification == frecStimulus[i]:
                complejo.append(1)
            else:
                complejo.append(0)
            
        magnitud.extend(complejo)
        datos.setdefault(f'Es{i}{s}', magnitud)
    '''
    Para que grafique todos los estimulos por sujeto
    
    Accuracy = {}
    
    for i in list(range(12)):
        AccureM = ((datos[f'Es{i}{s}'][0] + datos[f'Es{i}{s}'][1] + datos[f'Es{i}{s}'][2])/3)*100
        AccureC = ((datos[f'Es{i}{s}'][3] + datos[f'Es{i}{s}'][4] + datos[f'Es{i}{s}'][5])/3)*100
        Accuracy.setdefault(f'acc{i}{s}', [AccureM, AccureC])
        
    mag = []
    compl = []
    
    for i in list(range(12)):
        mag.append(Accuracy[f'acc{i}{s}'][0])
        compl.append(Accuracy[f'acc{i}{s}'][1])
        
    strings = []
    
    for i in list(range(12)):   
        strings.append(str(frecStimulus[i]))
        
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw = dict(hspace=0.45, wspace=0.2))
    
    plt.subplot(211)
    plt.bar(x = strings, height = mag, align = 'center', width = 0.8, color = 'skyblue')
    plt.xlabel("Frecuencia(Hz)")
    plt.ylabel('Accuracy (%)')
    plt.title(f'Sujeto {subjects[0]} Accuracy Magnitud')
    
    plt.subplot(212)
    plt.bar(x = strings, height = compl, align = 'center', width = 0.8, color = 'orange')
    plt.xlabel("Frecuencia(Hz)")
    plt.ylabel('Accuracy (%)')
    plt.title(f'Sujeto {s} Accuracy Complejos')
    
    plt.show()
  
    '''

Accuracy = {}

for i in list(range(12)):
    for s in subjects:
        AccureM = ((datos[f'Es{i}{s}'][0] + datos[f'Es{i}{s}'][1] + datos[f'Es{i}{s}'][2])/3)*100
        AccureC = ((datos[f'Es{i}{s}'][3] + datos[f'Es{i}{s}'][4] + datos[f'Es{i}{s}'][5])/3)*100
        Accuracy.setdefault(f'acc{i}{s}', [AccureM, AccureC])

strings = []
    
for i in subjects:   
    strings.append(f's{i}')

for i in list(range(12)):
    mag = []
    compl = []
    for s in subjects:
        mag.append(Accuracy[f'acc{i}{s}'][0])
        compl.append(Accuracy[f'acc{i}{s}'][1])
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw = dict(hspace=0.45, wspace=0.2))
    
    plt.subplot(211)
    plt.bar(x = strings, height = mag, align = 'center', width = 0.8, color = ['skyblue',"red","green","purple","orange","black","blue","cyan","magenta","yellow"])
    plt.xlabel("Frecuencia(Hz)")
    plt.ylabel('Accuracy (%)')
    plt.title(f'Estímulo {frecStimulus[i]} Accuracy Magnitud')
    
    plt.subplot(212)
    plt.bar(x = strings, height = compl, align = 'center', width = 0.8, color = ['skyblue',"red","green","purple","orange","black","blue","cyan","magenta","yellow"])
    plt.xlabel("Frecuencia(Hz)")
    plt.ylabel('Accuracy (%)')
    plt.title(f'Estímulo {frecStimulus[i]} Accuracy Complex')
    
    plt.show()
    

'''

Accuracy = {}

for i in list(range(12)):
    for s in subjects:
        AccureM = ((datos[f'Es{i}{s}'][0] + datos[f'Es{i}{s}'][1] + datos[f'Es{i}{s}'][2])/3)*100
        AccureC = ((datos[f'Es{i}{s}'][3] + datos[f'Es{i}{s}'][4] + datos[f'Es{i}{s}'][5])/3)*100
        Accuracy.setdefault(f'acc{i}{s}', [AccureM, AccureC])

strings = []
    
for i in subjects:   
    strings.append(f's{i}')

subplot = 211

fig, axes = plt.subplots(4, 3, figsize=(40, 28), gridspec_kw = dict(hspace=0.45, wspace=0.2))

for i in list(range(12)):
    mag = []
    compl = []
    for s in subjects:
        mag.append(Accuracy[f'acc{i}{s}'][0])
    
    plt.subplot(subplot)
    plt.bar(x = strings, height = mag, align = 'center', width = 0.8, color = ['skyblue',"red","green","purple","orange","black","blue","cyan","magenta","yellow"])
    plt.xlabel("Frecuencia(Hz)")
    plt.ylabel('Accuracy (%)')
    plt.title(f'Estímulo {frecStimulus[i]} Accuracy Magnitud')
    
    subplot = subplot +1
    
plt.show()
    
subplot = 211

fig, axes = plt.subplots(4, 3, figsize=(2, 1), gridspec_kw = dict(hspace=0.45, wspace=0.2))

for i in list(range(12)):
    compl = []
    for s in subjects:
        compl.append(Accuracy[f'acc{i}{s}'][1])
    
    plt.subplot(subplot)
    plt.bar(x = strings, height = compl, align = 'center', width = 0.8, color = ['skyblue',"red","green","purple","orange","black","blue","cyan","magenta","yellow"])
    plt.xlabel("Frecuencia(Hz)")
    plt.ylabel('Accuracy (%)')
    plt.title(f'Estímulo {frecStimulus[i]} Accuracy Complex')
    
    subplot = subplot +1
    
plt.show()
    
'''
    