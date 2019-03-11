# Reads wav files
# Converts wav to 1-D array (needs to assume wav was sampled on 1 channel)
# Splits wav into num_pieces
# Gets MFCC coefficients for each num_piece

# TODO
#  Use MFCC dataset as feature set
#  continue structuring dataset for training
#  build CNN model
#  Train
#  Evaluate

import argparse
import wave
import struct
import matplotlib.pyplot as plt
import csv
import numpy as np
from numpy import vstack

import scipy
from scipy import signal
from scipy.io.wavfile import read
from scipy.io import wavfile

import librosa
import librosa.display

from matplotlib import cm

from sklearn.preprocessing import minmax_scale # this one accepts 1D arrays
import scipy.stats as sp
import pandas as pd

def getWav(audio_wav):
    #upload .wav files (sampled from Windows side and copied over to guest side)
    rate, sampleWav0    = read(audio_wav)

    # gets one channel from the wav file input
    sample                                      = np.array(sampleWav0)

    dim = len(sample.shape)
    if dim >= 2:
        sampleWav = sampleWav0[:,1]
        sample = sample[:,1]
    else:
        sampleWav = sampleWav0

    return sample, sampleWav, rate

def splitWav(data, num_pieces):
    print("ORIGINAL SHAPE:", data.shape)
    a = data.reshape(num_pieces, int(len(data) / num_pieces))
    print("NEW SHAPE:", a.shape)
    return a

# I'm not the most familiar with MFCC and librosa, so at the moment I'm taking this piece for granted. 
# I'll question what exactly it does later, but on the surface it supposedly takes a time-domain waveform and 
# returns the MFFC coefficients.
#
# In this case, each one second sample (10000 samples) returns a 13x20 MFFC matrix
# I split one minute of sampling into 60 pieces, so I have 60 13x20 MFFC cofficients for each sampled "event"
#
# Generate mfccs from a time series

def getMFCC(data, rate, num_pieces):    
    mfcc = librosa.feature.mfcc(y=data[0], sr=rate, n_mfcc=13)
    mfcc = np.expand_dims(mfcc, axis=0)
    for i in range(1, num_pieces):
        result = librosa.feature.mfcc(y=data[i], sr=rate, n_mfcc=13)
        result = np.expand_dims(result, axis=0)
        mfcc = vstack((mfcc,result))    
    return mfcc

def plotData(left_d, right_d):
    # Subplots
    fig, axs = plt.subplots(2, 1)

    axs[1].plot(left_d)
    axs[0].plot(right_d)
    fig.tight_layout()
    plt.show()

def main():
    # two wav files.
    # left-side: me tapping on the left side of my computer for one minute
    # right-side: me tapping on the right side of my computer
    left_filename               = 'left-side.wav' #args.train_file    # get recorded wave
    right_filename              = 'right-side.wav' #args.predict_file

    # converting wav to numpy
    left_all, leftWav, rate     = getWav(left_filename)
    right_all, rightWav, rate   = getWav(right_filename)
        
    # normalize data
    left_scaled                 = minmax_scale(left_all)
    right_scaled                = minmax_scale(right_all)
    
    # reshape 1D array in MxN
    # I basically want to split my one minute of data into 60 one-second clips
    left_data                   = splitWav(left_scaled,60)
    right_data                  = splitWav(right_scaled,60)

    # MFCC features. I may have to explore this again later
    left_mfcc                   = getMFCC(left_data, rate, 60)
    right_mfcc                  = getMFCC(right_data, rate, 60)
       
    print("Final MFCC Shape:", left_mfcc.shape) # I want 60, MFCC matrices
    
    
    # --- LOOKING AT THINGS ---    
    #  maybe this isn't the best way to visualize differences ...
    #  but, I know I'll get 1 diagonally, but will get different values elsewhere
    #  I guess I'm thinking if the two mffc matrixs are different, I can "see" it this way
    #  If they are the same, or close, the graphs will be similar
    #  What I'm looking at clearly (visually) says "these are different"
    #  Since the two "look" different, I will go forward with using these MFCC matrices as features for a CNN
    #left_features               = pd.DataFrame(data=left_mfcc[0])
    #right_features              = pd.DataFrame(data=right_mfcc[0])
    #plt.matshow(left_features.corr())
    #plt.show()
    #plt.matshow(right_features.corr())
    #plt.show()
    #plotData(left_data[0], right_data[0])

if __name__ == '__main__':
    main()


