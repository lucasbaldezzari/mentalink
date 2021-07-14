# -*- coding: utf-8 -*-
"""
utils

Created on Sat May  8 10:31:22 2021

@author: Lucas
"""

import numpy as np
import matplotlib.pyplot as plt
import math

import os

from scipy.fftpack import fft, fftfreq
import scipy.fftpack

from scipy.signal import butter, filtfilt
    
def plotEEG(signal, sujeto = 1, trial = 1, blanco = 1,
            fm = 256.0, window = 1.0, rmvOffset = False, save = False, title = "", folder = "figs"):
    
    '''
    Grafica los canales de EEG pasados en la variable signal

    Argumentoss:
        - signal (numpy.ndarray): Arreglo con los datos de la señal
        - sujeto (int): Número se sujeto
        - trial (int): Trial a graficar
        - blanco (int): Algún blanco/target dentro del sistema de estimulación
        - fm (float): frecuencia de muestreo.
        - window (list): Valor mínimo y máximo -en segundos- que se graficarán
        - rmvOffset (bool): 

    Sin retorno:
    '''
    
    channelsNums = signal.shape[1]
    
    T = 1.0/fm #período de la señal
    
    totalLenght = signal.shape[2]
    beginSample = window[0]
    endSample = window[1]    

    #Chequeo que la ventana de tiempo no supere el largo total
    if beginSample/T >= totalLenght or beginSample <0:
        beginSample = 0.0 #muevo el inicio de la ventana a 0 segundos
        
    if endSample/T >= totalLenght:
        endSample = totalLenght*T #asigno el largo total
        
    if (endSample - beginSample) >0:
        lenght = (endSample - beginSample)/T #cantidad de valores para el eje x
        t = np.arange(1,lenght)*T + beginSample
    else:
        lenght = totalLenght
        t = np.arange(1,lenght)*T #máxima ventana
        
    scaling = 1 #(5/2**16) #supongo Vref de 5V y un conversor de 16 bits
    signalAvg = 0
    
    #genero la grilla para graficar
    fig, axes = plt.subplots(4, 2, figsize=(24, 20), gridspec_kw = dict(hspace=0.45, wspace=0.2))
    
    if not title:
        title = f"Señal de EEG de sujeto {sujeto}"
    
    fig.suptitle(title, fontsize=36)
    
    axes = axes.reshape(-1)
        
    for canal in range(channelsNums):
        if rmvOffset:
            # signalAvg = np.average(signal[target][canal-1].T[trial-1][:len(t)])
            signalAvg = np.average(signal[blanco - 1, canal - 1, :len(t), trial - 1])
        signalScale = (signal[blanco - 1, canal - 1, :len(t), trial - 1] - signalAvg)*scaling 
        axes[canal].plot(t, signalScale, color = "#e37165")
        axes[canal].set_xlabel('Tiempo [seg]', fontsize=16) 
        axes[canal].set_ylabel('Amplitud [uV]', fontsize=16)
        axes[canal].set_title(f'Sujeto {sujeto} - Blanco {blanco} - Canal {canal + 1}', fontsize=22)
        axes[canal].yaxis.grid(True)

    if save:
        pathACtual = os.getcwd()
        newPath = os.path.join(pathACtual, folder)
        os.chdir(newPath)
        plt.savefig(title, dpi = 600)
        os.chdir(pathACtual)
    
    plt.show()


def pasaBanda(canal, lfrec, hfrec, order, fm = 256.0, plot = False):
    '''
    Filtra la señal entre las frecuencias de corte lfrec (inferior) y hfrec (superior).
    Filtro del tipo "pasa banda"

    Argumentoss:
        - signal (numpy.ndarray): Arreglo con los datos de la señal
        - lfrec (float): frecuencia de corte baja (Hz).
        - hfrec (float): frecuencia de corte alta (Hz).
        - fm (float): frecuencia de muestreo (Hz).
        - orden (int): orden del filtro.

    Retorna:
        - canalFiltrado: canal filtrado en formato (numpy.ndarray)
        
        Info del filtro:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    '''
    
    nyquist = 0.5 * fm
    low = lfrec / nyquist
    high = hfrec / nyquist
    b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
    canalFiltrado = filtfilt(b, a, canal) #generamos filtro con los parámetros obtenidos
    
    return canalFiltrado
    
def filterEEG(signal, lfrec, hfrec, orden, fm = 256.0):
    '''
    Toma una señal de EEG y la filtra entre las frecuencias de corte lfrec (inferior) y hfrec (superior).

    Argumentos:
        - signal (numpy.ndarray): Arreglo con los datos de la señal
        - lfrec (float): frecuencia de corte baja (Hz).
        - hfrec (float): frecuencia de corte alta (Hz).
        - fm (float): frecuencia de muestreo (Hz).
        - orden (int): orden del filtro.

    Retorna:
        - retorna la señal filtrada de la forma (numpy.ndarray)[númeroClases, númeroCanales, númeroMuestras, númeroTrials]
    '''
    
    blancos = signal.shape[0]
    numCanales = signal.shape[1]
    totalLenght = signal.shape[2]
    numTrials = signal.shape[3]
    
    # signalFiltered: nparray para almacenar cada señal filtrada
    signalFiltered = np.zeros((signal.shape[0], signal.shape[1], totalLenght, signal.shape[3])) 

    #recorro la señal y aplico filtro pasabanda
    for blanco in range(0, blancos):
        for canal in range(0, numCanales):
            for trial in range(0, numTrials):
                signalFiltered[blanco, canal, : , trial] = pasaBanda(signal[blanco, canal, : ,trial],
                                                                     lfrec, hfrec, orden, fm)
    
    return signalFiltered


def ventaneo(data, duration, solapamiento):
    '''
    En base a la duración total de la muestra tomada y del solapamiento de los datos
    se devuelve la señal segmentada.
    
    Se recibe información de una sola clase, un solo canal y un solo trial 

    Args:
        - data (numpy.ndarray): muestras a ventanear 
        - duration (int): duración de la ventana, en cantidad de muestras.
        - solapamiento (int): cantidad de muestras a solapar

    Returns:
        - datosSegmentados
    '''
    
    segmentos = int(math.ceil((len(data) - solapamiento)/(duration - solapamiento)))
    
    tiempoBuf = [data[i:i+duration] for i in range(0, len(data), (duration - int(solapamiento)))]
    
    tiempoBuf[segmentos-1] = np.pad(tiempoBuf[segmentos-1],
                                    (0, duration - tiempoBuf[segmentos-1].shape[0]),
                                    'constant')
    
    datosSegmentados = np.vstack(tiempoBuf[0:segmentos])
    
    return datosSegmentados

def segmentingEEG(eeg, window, corriemiento, fm):
    '''
    Returns epoched eeg data based on the window duration and step size.

    Argumentoss:
        - eeg (numpy.ndarray): señal de eeg [Number of targets, Number of channels, Number of sampling points, Number of trials]
        - window (int): Duración de la ventana a aplicar (en segundos)
        - corriemiento (int): corriemiento de la ventana, en segundos.
        - fm (float): frecuencia de muestreo en Hz.

    Retorna:
        - Señal de EEG segmentada. 
        [targets, canales, trials, segments, duration]
    '''
    # Estructura de los datos en la señal de eeg
    # [Number of targets, Number of channels, Number of sampling points, Number of trials]
    
    clases = eeg.shape[0]
    channels = eeg.shape[1]
    samples = eeg.shape[2]
    trials = eeg.shape[3]

    
    duration = int(window * fm) #duración en cantidad de muestras
    solapamiento = (window - corriemiento) * fm #duración en muestras del solapamiento
    
    #se computa la cantidad de segmentos en base a la ventana y el solapamiento producido por el corrimiento
    segmentos = int(math.ceil((samples - solapamiento) / (duration - solapamiento)))
    
    segmentedEEG = np.zeros((clases, channels, trials, segmentos, duration))

    for target in range(0, clases):
        for channel in range(0, channels):
            for trial in range(0, trials):
                #Se agregan los segmentos a partir del ventaneo.
                segmentedEEG[target, channel, trial, :, :] = ventaneo(eeg[target, channel, :, trial], 
                                                                      duration, solapamiento) 

    return segmentedEEG


def computeMagnitudSpectrum(segmentedData, fftparms):
    '''
    Se computa la Transformada Rápida de Fourier a los datos pasados en segmentedData

    Argumentos:
        - segmentedData (numpy.ndarray): datos segmentados
        [targets, channels, trials, segments, samples].
        - fftparms (dict): dictionary of parameters used for feature extraction.
        - fftparms['resolución'] (float): resolución frecuencial
        - fftparms['frecuencia inicio'] (float): Componente frecuencial inicial en Hz
        - fftparms['frecuencia final'] (float): Componente frecuencial final en Hz 
        - fftparms['fm'] (float): frecuencia de muestreo en Hz.

    Retorna:
        - numpy.ndarray: Espectro de Fourier de la señal segmentedData
        [frecuency_componentes, num_channels, num_classes, num_trials, number_of_segments].
    '''
    
    targets = segmentedData.shape[0]
    channels = segmentedData.shape[1]
    trials = segmentedData.shape[2]
    segments = segmentedData.shape[3]
    fft_len = segmentedData[0, 0, 0, 0, :].shape[0]

    NFFT = round(fftparms['sampling_rate']/fftparms['resolution'])
    startIndex = int(round(fftparms['start_frequency']/fftparms['resolution']))
    endIndex = int(round(fftparms['end_frequency']/fftparms['resolution'])) + 1

    featuresData = np.zeros(((endIndex - startIndex),
                             channels, targets, trials, segments))
    
    #NOTA: Estos for anidados son muy demandantes y debe buscarse una alternativa, por ejemplo usando itertools
    for blanco in range(0, targets):
        for canal in range(0, channels):
            for trial in range(0, trials):
                for segment in range(0, segments):
                    FFT = np.fft.fft(segmentedData[blanco, canal, trial, segment, :], NFFT)/fft_len
                    espectro = 2*np.abs(FFT)
                    featuresData[:, canal, blanco, trial, segment] = espectro[startIndex:endIndex,]
    
    return featuresData

def computeComplexSpectrum(segmentedData, fftparms):
    '''
    Returns complex spectrum features. Fast Fourier Transform computed based on
    the FFT parameters provided as input. The real and imaginary parts of the input
    signal are concatenated into a single feature vector.

    Args:
        segmented_data (numpy.ndarray): epoched eeg data of shape 
        (num_classes, num_channels, num_trials, number_of_segments, num_samples)
        [targets, channels, trials, segments, samples].
        fftparms (dict): dictionary of parameters used for feature extraction.
        fftparms['resolution'] (float): frequency resolution per bin (Hz).
        fftparms['start_frequency'] (float): start frequency component to pick from (Hz). 
        fftparms['end_frequency'] (float): end frequency component to pick upto (Hz). 
        fftparms['sampling_rate'] (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): complex spectrum features of the input EEG.
        (2*n_fc, num_channels, num_classes, num_trials, number_of_segments)
    '''
    targets = segmentedData.shape[0]
    channels = segmentedData.shape[1]
    trials = segmentedData.shape[2]
    segments = segmentedData.shape[3]
    fft_len = segmentedData[0, 0, 0, 0, :].shape[0]
    
    NFFT = round(fftparms['sampling_rate']/fftparms['resolution'])
    startIndex = int(round(fftparms['start_frequency']/fftparms['resolution']))
    endIndex = int(round(fftparms['end_frequency']/fftparms['resolution'])) + 1
    
    featuresData = np.zeros((2*(endIndex - startIndex),
                             channels, targets, trials, segments))

    for target in range(0, targets):
        for channel in range(0, channels):
            for trial in range(0, trials):
                for segment in range(0, segments):
                    temp_FFT = np.fft.fft(segmentedData[target, channel, trial, segment, :], NFFT)/fft_len
                    real_part = np.real(temp_FFT)
                    imag_part = np.imag(temp_FFT)
                    featuresData[:, channel, target, trial, segment] = np.concatenate((
                        real_part[startIndex:endIndex,], 
                        imag_part[startIndex:endIndex,]), axis=0)
    
    return featuresData

def plotSpectrum(espectroSujeto, resol, blancos, sujeto, canal, frecStimulus,
                  startFrecGraph = 3.0, save = False, title = "", folder = "figs"):
    
    rows = len(frecStimulus)//4
    columns = len(frecStimulus)//3
    
    fig, plots = plt.subplots(columns, rows, figsize=(16, 14), gridspec_kw=dict(hspace=0.45, wspace=0.3))
    plots = plots.reshape(-1)
    
    if not title:
        title = f"Espectro de frecuecnias para canal {canal} - sujeto {sujeto}"
    
    fig.suptitle(title, fontsize = 20)
    
    for blanco in range(blancos):
        fft_axis = np.arange(espectroSujeto.shape[0]) * resol
        plots[blanco].plot(fft_axis + startFrecGraph,
                           np.mean(np.squeeze(espectroSujeto[:, canal, blanco, :, :]),
                                   axis=1), color = "#403e7d")
        plots[blanco].set_xlabel('Frecuencia [Hz]')
        plots[blanco].set_ylabel('Amplitud [uV]')
        plots[blanco].set_title(f'Estímulo {frecStimulus[blanco]}Hz del sujeto {sujeto}')
        plots[blanco].xaxis.grid(True)
        plots[blanco].axvline(x = frecStimulus[blanco], ymin = 0., ymax = max(fft_axis),
                             label = "Frec. Estímulo",
                             linestyle='--', color = "#e37165", alpha = 0.9)
        plots[blanco].legend()
        
    if save:
        pathACtual = os.getcwd()
        newPath = os.path.join(pathACtual, folder)
        os.chdir(newPath)
        plt.savefig(title, dpi = 500)
        os.chdir(pathACtual)
        
    plt.show()
    
def plotOneSpectrum(espectroSujeto, resol, blanco, sujeto, canal, frecStimulus,
                  startFrecGraph = 3.0, save = False, title = "", folder = "figs"):
    
    if not title:
        title = f"Spectrum for channel {canal} - sibject {sujeto}"
    
    # fig.suptitle(title, fontsize = 20)
    
    fft_axis = np.arange(espectroSujeto.shape[0]) * resol
    plt.plot(fft_axis + startFrecGraph,
                       np.mean(np.squeeze(espectroSujeto[:, canal, :, :, :]),
                               axis=1), color = "#403e7d")
    plt.xlabel('Frecuency [Hz]')
    plt.ylabel('Amplitud [uV]')
    plt.title(title)
    # plt.xaxis.grid(True)
    plt.axvline(x = frecStimulus, ymin = 0., ymax = max(fft_axis),
                         label = "Stimulus Frec.",
                         linestyle='--', color = "#e37165", alpha = 0.9)
    plt.legend()
        
    if save:
        pathACtual = os.getcwd()
        newPath = os.path.join(pathACtual, folder)
        os.chdir(newPath)
        plt.savefig(title, dpi = 500)
        os.chdir(pathACtual)
        
    plt.show()

