import os

import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import fileAdmin as fa
import pandas as pd

from tensorflow.keras.models import model_from_json

from scipy.signal import butter, filtfilt, windows
from scipy.signal import welch

from utils import filterEEG
import pickle

class CNNClassifier():
    
    def __init__(self, modelFile, weightFile, frecStimulus, nchannels,nsamples,ntrials,
                 PRE_PROCES_PARAMS, FFT_PARAMS, classiName = "", path = "models"):
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

        actualFolder = os.getcwd()#directorio donde estamos actualmente
        os.chdir(path)

        # load model from JSON file
        with open(f"{modelFile}.json", "r") as json_file:
            model = json_file.read()
            self.model = model_from_json(model)
            
        self.model.load_weights(f"{weightFile}.h5")
        self.model.make_predict_function()

        os.chdir(actualFolder)
        
        self.frecStimulus = frecStimulus
        self.nclases = len(frecStimulus)
        self.nchannels = nchannels #ex nchannels
        self.nsamples = nsamples #ex nsamples
        self.ntrials = ntrials #ex ntrials
        
        self.classiName = classiName #Classfier object name
        
        #Setting variables for EEG processing.
        self.PRE_PROCES_PARAMS = PRE_PROCES_PARAMS
        self.FFT_PARAMS = FFT_PARAMS

    def loadTrainingSignalPSD(self, filename = "", path = "models"):

        actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
        os.chdir(path)

        if not filename:
            filename = f'{self.modelName}_signalPSD.txt'
        self.trainingSignalPSD = np.loadtxt(filename, delimiter=',')
        
        os.chdir(actualFolder)

    def applyFilterBank(self, eeg, bw = 2.0, order = 4, calc1stArmonic = False):
        """Aplicamos banco de filtro a nuestros datos.
        Se recomienda aplicar un notch en los 50Hz y un pasabanda en las frecuencias deseadas antes
        de applyFilterBank()
        
        Args:
            - eeg: datos a aplicar el filtro. Forma [samples]
            - frecStimulus: lista con la frecuencia central de cada estímulo/clase
            - bw: ancho de banda desde la frecuencia central de cada estímulo/clase. Default = 2.0
            - order: orden del filtro. Default = 4"""

        nyquist = 0.5 * self.FFT_PARAMS["sampling_rate"]
        fcBanck = np.zeros((self.nclases,self.nsamples))

        for clase, frecuencia in enumerate(self.frecStimulus):   
            low = (frecuencia-bw/2)/nyquist
            high = (frecuencia+bw/2)/nyquist
            b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
            fcBanck[clase] = filtfilt(b, a, eeg) #filtramos

        if calc1stArmonic == True:
            firstArmonicBanck = np.zeros((self.nclases,self.nsamples))
            armonics = self.frecStimulus*2
            for clase, armonic in enumerate(armonics):   
                low = (armonic-bw/2)/nyquist
                high = (armonic+bw/2)/nyquist
                b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
                firstArmonicBanck[clase] = filtfilt(b, a, eeg) #filtramos

            aux = np.array((fcBanck, firstArmonicBanck))
            signalFilteredbyBank = np.sum(aux, axis = 0)

        else:
            signalFilteredbyBank = fcBanck #devuelvo señal filtrada solo en frecuencia central

        self.dataBanked = signalFilteredbyBank#.mean(axis = 0)
        return self.dataBanked

    def computWelchPSD(self, signalBanked, fm, ventana, anchoVentana, average = "median", axis = 1):
        self.signalSampleFrec, self.signalPSD = welch(signalBanked, fs = fm, window = ventana, nperseg = anchoVentana, average='median',axis = axis)

        return self.signalSampleFrec, self.signalPSD

    def pearsonFilter(self):
        """Lo utilizamos para extraer nuestro vector de características en base a analizar la correlación entre nuestro
        banco de filtro entrenado y el banco de filtro obtenido a partir de datos de EEG nuevos"""

        """
                    |Var(X) Cov(X,Y)|
        cov(X,Y) =  |               |
                    |Cov(Y,X) Var(Y)|
        """
        
        r_pearson = []
        for clase, frecuencia in enumerate(self.frecStimulus):
            covarianza = np.cov(self.trainingSignalPSD[clase], self.signalPSD)
            r_i = covarianza/np.sqrt(covarianza[0][0]*covarianza[1][1])
            r_pearson.append(r_i[0][1])

        indexFfeature = r_pearson.index(max(r_pearson))  

        return self.trainingSignalPSD[indexFfeature]

    def featuresExtraction(self, rawDATA, ventana, anchoVentana = 5, bw = 2.0, order = 4, axis = 1,
                            calc1stArmonic = False, usePearson = True):

        filteredEEG = filterEEG(rawDATA, self.PRE_PROCES_PARAMS["lfrec"],
                                self.PRE_PROCES_PARAMS["hfrec"],
                                self.PRE_PROCES_PARAMS["order"],
                                self.PRE_PROCES_PARAMS["bandStop"],
                                self.PRE_PROCES_PARAMS["sampling_rate"],
                                axis = axis)

        dataBanked = self.applyFilterBank(filteredEEG, bw=bw, order = 4, calc1stArmonic = calc1stArmonic)

        anchoVentana = int(self.PRE_PROCES_PARAMS["sampling_rate"]*anchoVentana) #fm * segundos
        ventana = ventana(anchoVentana)

        self.signalSampleFrec, self.signalPSD = self.computWelchPSD(dataBanked,
                                                fm = self.PRE_PROCES_PARAMS["sampling_rate"],
                                                ventana = ventana, anchoVentana = anchoVentana,
                                                average = "median", axis = 1)

        self.signalPSD = self.signalPSD.mean(axis = 0) #Obtengo el espectro promedio de cada espectro de la señal banqueada

        self.featureVector = self.pearsonFilter()

        if usePearson == True:
            self.featureVector = self.pearsonFilter() #selector de características

        else:
            self.featureVector = self.signalPSD

        numFeatures = self.featureVector.shape[0]
        self.featureVector = self.featureVector.reshape(1,1,numFeatures,1)

        return self.featureVector

    def getClassification(self, featureVector):
        """
        Method used to classify new data.
        
        Args:
            - dataForClassification: Data for classification. The shape must be
            []
        """
        self.preds = self.model.predict(featureVector)
        
        return self.frecStimulus[np.argmax(self.preds[0])]
        # return np.argmax(self.preds[0]) #máximo índice


def main():

    """Empecemos"""

    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"recordedEEG\WM\ses1")

    frecStimulus = np.array([6, 7, 8, 9])
    calc1stArmonic = False
    usePearson = False

    trials = 15
    fm = 200.
    window = 5 #sec
    samplePoints = int(fm*window)
    channels = 4

    filesRun1 = ["S3_R1_S2_E6","S3-R1-S1-E7", "S3-R1-S1-E8","S3-R1-S1-E9"]
    run1 = fa.loadData(path = path, filenames = filesRun1)
    filesRun2 = ["S3_R2_S2_E6","S3-R2-S1-E7", "S3-R2-S1-E8","S3-R2-S1-E9"]
    run2 = fa.loadData(path = path, filenames = filesRun2)

    def joinData(allData, stimuli, channels, samples, trials):
        joinedData = np.zeros((stimuli, channels, samples, trials))
        for i, sujeto in enumerate(allData):
            joinedData[i] = allData[sujeto]["eeg"][0,:,:,:trials]

        return joinedData #la forma de joinedData es [estímulos, canales, muestras, trials]

    run1JoinedData = joinData(run1, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)
    run2JoinedData = joinData(run2, stimuli = len(frecStimulus), channels = channels, samples = samplePoints, trials = trials)

    testSet = np.concatenate((run1JoinedData[:,:,:,12:], run2JoinedData[:,:,:,12:]), axis = 3) #últimos 3 tríals para testeo

    #### definimos archivos para cargar modelo posteriormente #### 
    #Abrimos archivos
    modelName = "cnntesting"
    modelFile = f"{modelName}.h5" #nombre del modelo
    PRE_PROCES_PARAMS, FFT_PARAMS = fa.loadPArams(modelName = modelName, path = os.path.join(actualFolder,"models"))

    descarteInicial = int(fm*PRE_PROCES_PARAMS['ti']) #en segundos
    descarteFinal = int(window*fm)-int(fm*PRE_PROCES_PARAMS['tf']) #en segundos
    
    testSet = testSet[:,:2, descarteInicial:descarteFinal ,:] #nos quedamos con los primeros dos canales y descartamos muestras iniciales y algunas finales

    testSet = np.mean(testSet, axis = 1) #promedio sobre los canales. Forma datos ahora [clases, samples, trials]

    nsamples = testSet.shape[1]
    ntrials = testSet.shape[2]

    #Restamos la media de la señal
    testSet = testSet - testSet.mean(axis = 1, keepdims=True)

    actualFolder = os.getcwd()#directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(actualFolder,"models")

    # Cargamos modelo previamente entrenado
    modefile = "cnntesting"
    path = os.path.join(actualFolder,"models")
    cnn = CNNClassifier(modelFile = modefile,
                    weightFile = "bestWeightss_cnntesting",
                    frecStimulus = frecStimulus.tolist(),
                    nchannels = 1,nsamples = nsamples,ntrials = ntrials,
                    PRE_PROCES_PARAMS = PRE_PROCES_PARAMS,
                    FFT_PARAMS = FFT_PARAMS,
                    classiName = f"CNN_Classifier", path = path)

    cnn.loadTrainingSignalPSD(filename = "cnntesting_signalPSD.txt", path = path) #cargamos el PSD de mis datos de entrenamiento

    anchoVentana = (window - PRE_PROCES_PARAMS['ti'] - PRE_PROCES_PARAMS['tf']) #fm * segundos
    ventana = windows.hamming

    clase = 2
    trial = 6

    rawDATA = testSet[clase-1,:,trial-1]

    #extrameos características
    featureVector  = cnn.featuresExtraction(rawDATA = rawDATA, ventana = ventana, anchoVentana = anchoVentana,
                        bw = 2.0, order = 4, axis = 0, calc1stArmonic = calc1stArmonic, usePearson=usePearson)

    cnn.getClassification(featureVector = featureVector)

    ### Realizamos clasificación sobre mis datos de testeo. Estos nunca fueron vistos por el clasificador ###
    trials = 6 #cantidad de trials
    predicciones = np.zeros((len(frecStimulus),trials)) #donde almacenaremos las predicciones

    for i, clase in enumerate(np.arange(len(frecStimulus))):
        for j, trial in enumerate(np.arange(trials)):
            rawDATA = testSet[clase, :, trial]
            featureVector  = cnn.featuresExtraction(rawDATA = rawDATA, ventana = ventana,
                            anchoVentana = anchoVentana, bw = 1.0, order = 4, axis = 0,
                            calc1stArmonic = calc1stArmonic, usePearson=usePearson)

            classification = cnn.getClassification(featureVector = featureVector)
            if classification == frecStimulus[clase]:
                predicciones[i,j] = 1

        predicciones[i,j] = predicciones[i,:].sum()/trials

    predictions = pd.DataFrame(predicciones, index = frecStimulus,
                    columns = [f"trial {trial+1}" for trial in np.arange(trials)])

    predictions['promedio'] = predictions.mean(numeric_only=True, axis=1)
    
    print(f"Predicciones usando el modelo SVM {modefile}")
    print(predictions)

if __name__ == "__main__":
    main()



