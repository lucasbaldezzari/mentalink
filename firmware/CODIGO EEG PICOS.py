# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:01:15 2021

@author: enfil
"""

import fileAdmin as fa
from utils import plotEEG, segmentingEpochs, plotSpectrum, filterEEG, getSpectrum, pasaBanda
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

path = "C:/Users/enfil/Documents/GitHub/mentalink/talleres/Taller_2/scripts/dataset"
sujeto8 = fa.loadData(path,[8]) #es un diccionario, donde unicamente importe los datos del sujeto 8

print(type(sujeto8["s8"])) #tipo de datos del sujeto 8

print(sujeto8["s8"].keys()) #imprimimos las llaves del diccionario

print(sujeto8["s8"]["eeg"].shape) #tamaño de la matriz de datos en la key eeg del sujeto 8 (matriz)

"""Desde aqui modificas los datos del sujeto que quieras"""

sujeto = 8 #quiero trabajar sobre los datos del sujeto 8
eeg = sujeto8[f"s{sujeto}"]["eeg"] #matriz de datos del sujeto 8, guardados en "eeg"
clases = eeg.shape[0] #clases del sujeto 1
channels = eeg.shape[1] #canales del sujeto 
samples= eeg.shape[2] #cantidad de muestras
trials= eeg.shape[3] # cantidad de trials

target = 5 #elijo los datos viendo el tarjet 5

fm = 256.0
T = 1/fm
data = []
FFTe = []
frecs = []

eeg = eeg[target][1] #cargo los datos del tarjet 5 canal 1

tiempo = np.arange(0,len(eeg)*T,T) #valores en el eje de las x para la señal de eeg vs tiempo

for i in range(len(eeg)): 
    data.append(eeg[i][0]) #en una lista ingresamos todos los datos del 
                             #primer trial de cada sampling point

array = np.array(data) #convertimos la lista en array para hacerle la FFT

largo = len(data)
frecuencias = np.arange(0, fm, (fm/largo)) #frecuencias para el eje x de la FFT

array = pasaBanda(array, 5, 40, 6, fm = 256.0) #filtramos la señal

FFT = fft(array) #realiza la transformada de los datos

for i in range(125): #restringue el rango de frecuencia que se muestra, si range(1114) son 256 Hz
    FFTe.append(FFT[i])
    frecs.append(frecuencias[i])

FFTe = np.array(FFTe)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw = dict(hspace=0.4, wspace=0.2))

plt.subplot(211)
plt.plot(tiempo, array)
plt.xlabel("tiempo(s)")
plt.ylabel("Voltaje(uV)")
plt.suptitle("Voltaje vs tiempo")

plt.subplot(212)
plt.plot(frecs, abs(FFTe))
plt.xlabel("Frecuencia(Hz)")
plt.ylabel("Amplitud")
plt.title("Amplitud vs frecuencia")
plt.show()
    
a = list(FFTe)
a.append(1)
umbral = 1500

for i in range(len(a)): #codigo para detecta maximos sobre umbrales
    if abs(a[i+2]) ==1:
        break
    if abs(a[i+1]) > abs(a[i]) and abs(a[i+1]) >= abs(a[i+2]) and abs(a[i+1]) > umbral:
        print(f"hay un maximo en {frecuencias[i+1]} hz")
