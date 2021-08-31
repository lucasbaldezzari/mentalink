# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:29:54 2021

@author: enfil
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

Amp = 1
freq = 8
freqs = 48
phip = 0
tiempo = 2 
lista = []
herzio = []
umbral = 40

def sin_wave(A, f, fs, phi, t):
    '''
    : Params A: Amplitud
         : params f: Frecuencia de señal
         : Params fs: Frecuencia de muestreo
         : Params phi: fase
         : params t: longitud de tiempo
    '''
    # Si la longitud de la serie de tiempo es T = 1S, 
    # Frecuencia de muestreo FS = 1000 Hz, intervalo de tiempo de muestreo TS = 1 / FS = 0.001S
    # Para los puntos de muestreo de la secuencia de tiempo es n = t / t ts = 1 / 0,001 = 1000, hay 1000 puntos, cada intervalo de punto es TS
    Ts = 1/fs
    n = t / Ts
    n = np.arange(n)
    y = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
    return y

señal = sin_wave(Amp, freq, freqs, phip, tiempo)
largo = len(señal)
ejex = np.arange(0, tiempo, 1/freqs)

transf = fft(señal)
largo
frecuencias = np.arange(0, freqs, (freqs/largo))

for i in range(freqs):
    lista.append(transf[i])
    herzio.append(frecuencias[i])
    
a = np.array(lista)
lista.append(1)

for i in range(len(lista)): #codigo para detecta maximos sobre umbrales
    #print(abs(lista[i]), abs(lista[i+1]), abs(lista[i+2]))
    if abs(lista[i+2]) ==1:
        break
    if abs(lista[i+1]) > abs(lista[i]) and abs(lista[i+1]) >= abs(lista[i+2]) and abs(lista[i+1]) > umbral:
        print(f"hay un maximo en {frecuencias[i+1]} hz")
    
plt.figure()
plt.subplot(211)
plt.plot(ejex, señal)

plt.subplot(212)
plt.plot(herzio, abs(a))
plt.show()