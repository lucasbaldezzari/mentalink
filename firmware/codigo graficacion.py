"Este codigo grafica la señal de EEG sin filtrar y filtrada de cada sujeto en dataset"
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:53:14 2021

@author: enfil
"""

import fileAdmin as fa

#import numpy as np

#import matplotlib.pyplot as plt

from utils import plotEEG
from utils import filterEEG
#from utils import segmentingEpochs, getSpectrum
#from utils import plotSpectrum
# from utils import segmented_data, magnitude_spectrum
# from utils import plotSpectrumEEG

#from scipy.fftpack import fft, fftfreq
#import scipy.fftpack

path = "C:/Users/enfil/Documents/GitHub/mentalink/talleres/Taller_2/scripts/dataset" #directorio donde estan los datos
sujetos = [1,2,3,4,5,6,7,8] #sujetos 1 y 2

setSubjects = fa.loadData(path = path, subjects = sujetos)

for i in sujetos:
    print(f'Paciente {i}, Blanco 5')
    #Conociendo mis datos
    print(type(setSubjects[f's{i}'])) #tipo de datos del sujeto 1
    
    print(setSubjects[f's{i}'].keys()) #imprimimos las llaves del diccionario
    
    print(setSubjects[f's{i}']["eeg"].shape) #imprimimos la forma del dato en "eeg"
    # Obtenemos un arreglo que se corresponde con lo mencionado en la referencia
    # [Number of targets, Number of channels, Number of sampling points, Number of trials]
    
    
    """Grafiquemos los datos obtenidos para todos los sujetos en los 8 canales y el blanco de 9.25Hz"""
    
    sujeto = i
    eeg = setSubjects[f"s{sujeto}"]["eeg"]
    clases = eeg.shape[0] #clases del sujeto 1
    channels = eeg.shape[1] #canales del sujeto 
    samples= eeg.shape[2] #cantidad de muestras
    trials= eeg.shape[3] # cantidad de trials
    
    target = 5 #el máximo es 12
    
    fm = 256.0
    
    """
    Los autores del set de datos, citan:
        The onset of visual stimulation is at 39th sample point,
        which means there are redundant data for 0.15 [s] before stimulus onset.
        
    Con esto en mente es que se descartan las primeras 39 muestras
    """
    
    tiempoTotal = int(4*fm) #cantidad de muestras para 4segundos
    muestraDescarte = 39
    
    eeg = eeg[:,:, muestraDescarte: ,:]
    eeg = eeg[:,:, :tiempoTotal ,:]
    
    title = f"Señal de EEG sin filtro sujeto {sujeto}"
    plotEEG(signal = eeg, sujeto = sujeto,
            trial = 3, blanco = target, window = [0,4], fm = 256.0, save = False, title = title)
    
    #filtro la señal entre los 5hz y los 80hz
    eegfiltrado = filterEEG(eeg, lfrec = 5., hfrec = 80., orden = 4, fm  = 256.0)
    
    title = f"Señal de EEG filtrada sujeto {sujeto}"
    
    plotEEG(eegfiltrado, sujeto = sujeto,
            trial = 3, blanco = target, window = [0,4], fm = 256.0, save = False,
            title = title)