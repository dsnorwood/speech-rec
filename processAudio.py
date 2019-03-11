# Reads wav files
# Converts wav to 1-D array (needs to assume wav was sampled on 1 channel)
# FFT of converted wav
# sublot three waveforms: FFT, time, spectrogram

# TODO
#  generate MFCC
#  create dataset with key features
#  classify dataset as GOOD/BAD

from __future__ import print_function

import argparse
import wave
import struct
import wav2vec
import matplotlib.pyplot as plt
import csv
import numpy as np

import scipy
from scipy import signal
from scipy.io.wavfile import read
from scipy.io import wavfile

import librosa
import librosa.display

from matplotlib import cm

from keras.models import Sequential  
from keras.layers import Dense, Flatten, Activation  
from keras.layers import LSTM
from keras.callbacks import TensorBoard


def buildLSTM():
    print("building network ... ")

    model = Sequential() 
    model.add(Dense(200, input_dim=1130, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) 
#   model.add(LSTM(hidden_neurons, return_sequences=False, input_shape=[1,one_d_features.size]))
#   model.add(Flatten())
#   model.add(Dense(hidden_neurons, input_dim=in_out_neurons))
#   model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop") 

    return model

def buildTrainingData(good, bad):
    print("buliding training data ... ")

    # concate multiple 1D arrays into multidimensional array
    training_set   = np.vstack((good, good, bad, bad, good, bad, good, good, good, bad))
    training_data  = np.vstack((training_set, training_set, training_set, training_set, training_set, training_set, training_set, training_set, training_set, training_set))

    #input and classifier
    raw             = training_data[:,0:1130]
    classifiers     = training_data[:,1130]

    print("training_data:")
    print(training_data.shape)
#   print(training_data)
#   print("raw:") # data without classifiers
#   print(raw.shape)
#   print(raw)
#   print("classifiers:") # only classifiers
#   print(classifiers.shape)
#   print(classifiers)

    return training_data, raw, classifiers

def addClassifiers(raw_features, classifier):
    print("adding classifier to training data ... ")

#   # EXAMPLE
#   dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
#   # split into input (X) and output (Y) variables
#   X = dataset[:,0:8]
#   Y = dataset[:,8]
#
#   print("DATASET:") # dataset + classifiers
#   print(dataset.shape)
#   print(dataset)
#   print("X:") # data without classifiers
#   print(X.shape)
#   print(X)
#   print("Y:") # only classifiers
#   print(Y.shape)
#   print(Y)

    # add classifier
    #print(raw_features)
    classified_features = raw_features
    classified_features[1130] = classifier
    #print(classified_features)
#
#   # concate multiple 1D arrays into multidimensional array
#   features=np.vstack((one_d_features, one_d_features, one_d_features))
#   print(features)
#   print(features.shape)
#
#   #input and classifier
#   X = features[:,0:1130]
#   Y = features[:,1130]
#   print(X.shape)
#   print(Y.shape)

    return classified_features

def trainLSTM(model, raw, classifiers):
    print("training network ... ")
    
    model.fit(raw, classifiers, epochs=150, batch_size=10, verbose=0)
    #keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,write_graph=True, write_images=True)
    #model.fit(...inputs and parameters..., callbacks=[tbCallBack])
    scores = model.evaluate(raw, classifiers)
    print("SCORES/NAMES:")
    print(scores)
    print(model.metrics_names)
    #print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    return model

def predictLSTM(model, raw):
    print("making a prediction ... ")
    print("DATA_IN:")
    print(raw.shape)
    predictions = model.predict(raw)
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    print(rounded)


def generateTrainingSet(raw_features):
    print("generating training set ... ")
    #print(features.shape)
    #print(raw_features)
    reshaped_features = raw_features.ravel()
    #print(reshaped_features)

    return reshaped_features

def getMFCC(filename):
    print("getting features ... ")
    y, sr = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # METHOD 1
#   S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
#   librosa.feature.mfcc(S=librosa.power_to_db(S))
#   mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#   return mfccs
#   plt.figure(figsize=(10, 4))
#   librosa.display.specshow(mfccs, x_axis='time')
#   plt.colorbar()
#   plt.title('MFCC')
#   plt.tight_layout()
#   plt.show()

    # METHOD 2
#   ig, ax = plt.subplots()
#   mfcc_data= np.swapaxes(mfcc, 0 ,1)
#   cax = ax.imshow(mfcc, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
#   ax.set_title('MFCC')
#   #Showing mfcc_data
#   plt.show()
#   #Showing mfcc_feat
#   plt.plot(mfcc)
#   plt.show()

    return mfcc

def getWav(audio_wav):
    print("getting wav file specs ... ")
    #upload .wav files (sampled from Windows side and copied over to guest side)
    rate, sampleWav0 	= read(audio_wav)

    # gets one channel from the wav file input
    sample 					= np.array(sampleWav0)

    dim = len(sample.shape)
    if dim >= 2:
        sampleWav = sampleWav0[:,0]
        sample = sample[:,0]
    else:
        sampleWav = sampleWav0

    return sample, sampleWav, rate
    
def getTestWav():
    test_filename = "mysine.wav"
    t = 1
    fs = 16000
    f = 100
    
    samples = np.linspace(0, t, int(fs*t), endpoint=False)
    signal = np.sin(2 * np.pi * f * samples)
    signal *= 32767
    signal = np.int16(signal)

    # PLOT TIME DOMAIN
    #plt.plot(signal)
    #plt.show()

    wavfile.write(test_filename, fs, signal)
    return test_filename

def getFFT(sample):
    print("getting FFT ... ")
    SAMPLE 					= np.fft.fft(sample)
    return SAMPLE

def plotData(SAMPLE, sample, sample_mfcc):
    # Subplots
    fig, axs 		        = plt.subplots(3, 1)

    axs[0].plot(np.abs(SAMPLE))
    axs[1].plot(sample)
    librosa.display.specshow(sample_mfcc, x_axis='time')

    fig.tight_layout()
    plt.show()

def main():
    parser          = argparse.ArgumentParser()
    parser.add_argument('--train_file', '-t', type=str)
    parser.add_argument('--predict_file', '-p', type=str)
    args            = parser.parse_args()
    
    #filename       = getTestWav() # get test wave
    train_filename        = args.train_file    # get recorded wave
    predict_filename      = args.predict_file

    train, trainWav, rate       = getWav(train_filename) 
    predict, predictWav, rate   = getWav(predict_filename) 

    # for plotting
    TRAIN           = getFFT(train)
    PREDICT         = getFFT(predict)

    # features for network
    train_mfcc      = getMFCC(train_filename)
    predict_mfcc    = getMFCC(predict_filename)

    #plotData(TRAIN, train, train_mfcc)
    #plotData(PREDICT, predict, predict_mfcc)

    post_train      = generateTrainingSet(train_mfcc)
    post_predict    = generateTrainingSet(predict_mfcc)

    classified_train   = addClassifiers(post_train, 1)
    classified_predict = addClassifiers(post_predict, 0)

    training_data, raw, classifiers = buildTrainingData(classified_train, classified_predict)

    print(training_data.shape)
    print(raw.shape)
    print(classifiers.shape)

    model           = buildLSTM()
    model           = trainLSTM(model, raw, classifiers)

    print("CLASSIFIED TRAIN:")
    print(classified_train.shape)
    print(classified_train)

    raw = classified_train[:1130]
    print("RAW:")
    print(raw.shape)
    print(raw)

    predictLSTM(model, raw)

if __name__ == '__main__':
    main()
