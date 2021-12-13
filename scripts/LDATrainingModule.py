"""LDATrainingModule V1.0"""


import os
import numpy as np
import numpy.matlib as npm
import json

from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from scipy.signal import butter, filtfilt, windows
from scipy.signal import welch

import pickle

import matplotlib.pyplot as plt

import fileAdmin as fa
from utils import filterEEG

class LDATrainingModule():

    def __init__(self, rawDATA, PRE_PROCES_PARAMS, FFT_PARAMS, frecStimulus, nchannels,nsamples,ntrials,modelName = ""):
        """Variables de configuración

        Args:
            - rawDATA(matrix[clases x canales x samples x trials]): Señal de EEG
            - subject (string o int): Número de sujeto o nombre de sujeto
            - PRE_PROCES_PARAMS: Parámetros para preprocesar los datos de EEG
            - FFT_PARAMS: Parametros para computar la FFT
            - modelName: Nombre del modelo
        """

        self.rawDATA = rawDATA

        if not modelName:
            self.modelName = f"LDAModel"

        else:
            self.modelName = modelName

        self.frecStimulus = frecStimulus
        self.nclases = len(frecStimulus)
        self.nchannels = nchannels
        self.nsamples = nsamples
        self.ntrials = ntrials

        self.dataBanked = None #datos de EEG filtrados con el banco
        self.model = None
        self.trainingData = None
        self.labels = None

        self.signalPSD = None #PSD de mis datos
        self.signalSampleFrec = None

        self.MSF = np.array([]) #Magnitud Spectrum Features

        self.PRE_PROCES_PARAMS = PRE_PROCES_PARAMS
        self.FFT_PARAMS = FFT_PARAMS

        self.METRICAS = {f'modelo_{self.modelName}': {'trn': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                                                      'val': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None}}}

    def createLDA(self, solver = "svd", shrinkage = None):
        """Se crea modelo"""

        self.model = LinearDiscriminantAnalysis(solver = solver, shrinkage = shrinkage)

        return self.model

    def applyFilterBank(self, eeg, bw = 2.0, order = 4, axis = 0, calc1stArmonic = False):
        """Aplicamos banco de filtro a nuestros datos.
        Se recomienda aplicar un notch en los 50Hz y un pasabanda en las frecuencias deseadas antes
        de applyFilterBank()
        
        Args:
            - eeg: datos a aplicar el filtro. Forma [clase, samples, trials]
            - frecStimulus: lista con la frecuencia central de cada estímulo/clase
            - bw: ancho de banda desde la frecuencia central de cada estímulo/clase. Default = 2.0
            - order: orden del filtro. Default = 4"""

        nyquist = 0.5 * self.FFT_PARAMS["sampling_rate"]
        #signalFilteredbyBank = np.zeros((self.nclases,self.nsamples,self.ntrials))
        fcBanck = np.zeros((self.nclases,self.nsamples,self.ntrials))
        firstArmonicBanck = np.zeros((self.nclases,self.nsamples,self.ntrials))

        for clase, frecuencia in enumerate(self.frecStimulus):   
            low = (frecuencia-bw/2)/nyquist
            high = (frecuencia+bw/2)/nyquist
            b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
            fcBanck[clase] = filtfilt(b, a, eeg[clase], axis = axis) #filtramos

        if calc1stArmonic == True:
            firstArmonicBanck = np.zeros((self.nclases,self.nsamples,self.ntrials))
            armonics = self.frecStimulus*2
            for clase, armonic in enumerate(armonics):   
                low = (armonic-bw/2)/nyquist
                high = (armonic+bw/2)/nyquist
                b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
                firstArmonicBanck[clase] = filtfilt(b, a, eeg[clase], axis = axis) #filtramos

            aux = np.array((fcBanck, firstArmonicBanck))
            signalFilteredbyBank = np.sum(aux, axis = 0)

        else:
            signalFilteredbyBank = fcBanck

        self.dataBanked = signalFilteredbyBank

        return self.dataBanked

    def computWelchPSD(self, signalBanked, fm, ventana, anchoVentana, average = "median", axis = 1):

        self.signalSampleFrec, self.signalPSD = welch(signalBanked, fs = fm, window = ventana, nperseg = anchoVentana, average='median',axis = axis)

        return self.signalSampleFrec, self.signalPSD


    def featuresExtraction(self, ventana, anchoVentana = 5, bw = 2.0, order = 4, axis = 1, calc1stArmonic = False):

        filteredEEG = filterEEG(self.rawDATA, self.PRE_PROCES_PARAMS["lfrec"],
                                self.PRE_PROCES_PARAMS["hfrec"],
                                self.PRE_PROCES_PARAMS["order"],
                                self.PRE_PROCES_PARAMS["bandStop"],
                                self.PRE_PROCES_PARAMS["sampling_rate"],
                                axis = axis)

        dataBanked = self.applyFilterBank(filteredEEG, bw=bw, order = 4, calc1stArmonic = calc1stArmonic) #Aplicamos banco de filtro

        anchoVentana = int(self.PRE_PROCES_PARAMS["sampling_rate"]*anchoVentana) #fm * segundos
        ventana = ventana(anchoVentana)

        self.signalSampleFrec, self.signalPSD = self.computWelchPSD(dataBanked,
                                                fm = self.PRE_PROCES_PARAMS["sampling_rate"],
                                                ventana = ventana, anchoVentana = anchoVentana,
                                                average = "median", axis = axis)

        return self.signalSampleFrec, self.signalPSD

    def trainAndValidateLDA(self, clases, test_size = 0.2, randomSeed = None):
        """Método para entrenar un modelo LDA.

        Argumentos:
            - clases (int): Lista con valores representando la cantidad de clases
            - test_size: Tamaño del set de validación"""

        self.frecStimulus = clases

        self.trainingData, self.labels = self.getDataForTraining(self.signalPSD)

        X_trn, X_val, y_trn, y_val = train_test_split(self.trainingData, self.labels, test_size = test_size, shuffle = True, random_state = randomSeed)

        self.model.fit(X_trn,y_trn)

        y_pred = self.model.predict(X_trn)

        precision, recall, f1,_ = precision_recall_fscore_support(y_trn, y_pred, average='weighted')

        self.METRICAS = {f'modelo_{self.modelName}': {'trn': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                                                      'val': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None}}}

        precision, recall, f1,_ = precision_recall_fscore_support(y_trn, y_pred, average='macro')
        accuracy = accuracy_score(y_trn, y_pred)


        self.METRICAS[f'modelo_{self.modelName}']['trn']['Pr'] = precision
        self.METRICAS[f'modelo_{self.modelName}']['trn']['Rc'] = recall
        self.METRICAS[f'modelo_{self.modelName}']['trn']['Acc'] = accuracy
        self.METRICAS[f'modelo_{self.modelName}']['trn']['F1'] = f1

        y_pred = self.model.predict(X_val)

        precision, recall, f1,_ = precision_recall_fscore_support(y_val, y_pred, average='macro')
        accuracy = accuracy_score(y_val, y_pred)

        self.METRICAS[f'modelo_{self.modelName}']['val']['Pr'] = precision
        self.METRICAS[f'modelo_{self.modelName}']['val']['Rc'] = recall
        self.METRICAS[f'modelo_{self.modelName}']['val']['Acc'] = accuracy
        self.METRICAS[f'modelo_{self.modelName}']['val']['F1'] = f1

        return self.METRICAS

    #Transforming data for training
    def getDataForTraining(self, features):
        """Preparación del set de entrenamiento.

        Argumentos:
            - features: Parte Real del Espectro or Parte Real e Imaginaria del Espectro
            con forma [número de características x canales x clases x trials x número de segmentos]
            - clases: Lista con las clases para formar las labels

        Retorna:
            - trainingData: Set de datos de entrenamiento para alimentar el modelo LDA
            Con forma [trials*clases x number of features]
            - Labels: labels para entrenar el modelo a partir de las clases
        """

        numFeatures = features.shape[1]
        trainingData = features.swapaxes(2,1).reshape(self.nclases*self.ntrials, numFeatures)

        classLabels = np.arange(self.nclases)

        labels = (npm.repmat(classLabels, self.ntrials, 1).T).ravel()

        return trainingData, labels


    def saveTrainingSignalPSD(self, signalPSD, path, filename = ""):

        actualFolder = os.getcwd()#directorio donde estamos actualmente
        os.chdir(path)
        
        if not filename:
            filename = self.modelName

        np.savetxt(f'{filename}_signalPSD.txt', signalPSD, delimiter=',')
        np.savetxt(f'{filename}_signalSampleFrec.txt', self.signalSampleFrec, delimiter=',')

        os.chdir(actualFolder)
        
    def saveModel(self, path, filename = ""):
        """Método para guardar el modelo"""

        actualFolder = os.getcwd()#directorio donde estamos actualmente
        os.chdir(path)

        if not filename:
            modelName = self.modelName
            filename = f"{self.modelName}.pkl"

        else:
            modelName = filename
            filename = f"{filename}.pkl"

        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

        #Guardamos los parámetros usados para entrenar el SVM
        file = open(f"{modelName}_preproces.json", "w")
        json.dump(self.PRE_PROCES_PARAMS , file)
        file.close

        file = open(f"{modelName}_fft.json", "w")
        json.dump(self.FFT_PARAMS , file)
        file.close

        os.chdir(actualFolder)


def main():

    """Empecemos"""

    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"recordedEEG\WM\ses1")

    frecStimulus = np.array([6, 7, 8, 9])
    calc1stArmonic = True 

    trials = 15
    fm = 200.
    window = 5 #sec
    samplePoints = int(fm*window)
    channels = 4

    #Seteamos parámetros para 
    ti = 0.5 #en segundos
    tf = 0.5 #en segundos
    descarteInicial = int(fm*ti) #en segundos
    descarteFinal = int(window*fm)-int(tf*fm) #en segundos

    filesRun1 = ["S3_R1_S2_E6","S3-R1-S1-E7", "S3-R1-S1-E8","S3-R1-S1-E9"]
    run1 = fa.loadData(path = path, filenames = filesRun1)
    filesRun2 = ["S3_R2_S2_E6","S3-R2-S1-E7", "S3-R2-S1-E8","S3-R2-S1-E9"]
    run2 = fa.loadData(path = path, filenames = filesRun2)

    #Filtering de EEG
    PRE_PROCES_PARAMS = {
                    'lfrec': 4.,
                    'hfrec': 38.,
                    'order': 8,
                    'sampling_rate': fm,
                    'bandStop': 50.,
                    'window': window,
                    'shiftLen':window,
                    'ti': ti, 'tf':tf
                    }

    resolution = np.round(fm/samplePoints, 4)

    FFT_PARAMS = {
                    'resolution': resolution,#0.2930,
                    'start_frequency': 4.0,
                    'end_frequency': 38.0,
                    'sampling_rate': fm
                    }

    def joinData(allData, stimuli, channels, samples, trials):
        joinedData = np.zeros((stimuli, channels, samples, trials))
        for i, sujeto in enumerate(allData):
            joinedData[i] = allData[sujeto]["eeg"][0,:,:,:trials]

        return joinedData #la forma de joinedData es [estímulos, canales, muestras, trials]

    run1JoinedData = joinData(run1, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)
    run2JoinedData = joinData(run2, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)

    trainSet = np.concatenate((run1JoinedData[:,:,:,:12], run2JoinedData[:,:,:,:12]), axis = 3)
    trainSet = trainSet[:,:2, descarteInicial:descarteFinal,:] #nos quedamos con los primeros dos canales y descartamos muestras iniciales y algunas finales

    trainSet = np.mean(trainSet, axis = 1) #promedio sobre los canales. Forma datos ahora [clases, samples, trials]

    nsamples = trainSet.shape[1]
    ntrials = trainSet.shape[2]

    #Restamos la media de la señal
    trainSet = trainSet - trainSet.mean(axis = 1, keepdims=True)

    lda = LDATrainingModule(trainSet, PRE_PROCES_PARAMS, FFT_PARAMS, frecStimulus=frecStimulus,
    nchannels = 1, nsamples = nsamples, ntrials = ntrials, modelName = "LDAtest")
    
    seed = np.random.randint(100)

    modelo = lda.createLDA(solver = "eigen", shrinkage = "auto")
    # lda.model

    anchoVentana = (window - ti - tf) #fm * segundos
    ventana = windows.hamming

    sampleFrec, signalPSD  = lda.featuresExtraction(ventana = ventana, anchoVentana = anchoVentana, bw = 2.0, order = 4, axis = 1, calc1stArmonic = calc1stArmonic)

    trainingData, labels = lda.getDataForTraining(signalPSD)

    metricas = lda.trainAndValidateLDA(clases = np.arange(0,len(frecStimulus)), test_size = 0.2, randomSeed = seed)
    print("**** METRICAS ****")
    print(metricas)
    
    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"models")
    lda.saveModel(path)
    lda.saveTrainingSignalPSD(signalPSD.mean(axis = 2), path = path, filename = "LDA_WM_testing")
    os.chdir(actualFolder)

    
if __name__ == "__main__":
    main()