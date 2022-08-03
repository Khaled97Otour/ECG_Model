#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv1D, BatchNormalization,MaxPool1D, GlobalMaxPool1D
from keras.layers import TimeDistributed, GRU, Dense, Dropout, Masking, Embedding, LSTM ,Flatten
from keras.layers import Input, Conv1D, DepthwiseConv1D,GlobalMaxPool1D,      Dense, Concatenate, Add, ReLU, BatchNormalization, AvgPool1D,MaxPool1D,      GlobalAvgPool1D, Reshape, Permute, Lambda, Activation,RepeatVector
from keras import layers
from keras import models
import keras.backend as K
from keras.models import Model
import keras.backend as K
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy, matplotlib
import csv
from scipy.fft import ifft,rfft

def Import_data(x,y):
  # read the files and build a dataset 
  # after processing the input signals
  rows = []
  rows1= []
  for row in x:
    rows.append(row)
  for row1 in y:
    rows1.append(row1)
  raw_normal_signals=np.asarray(rows)
  raw_Abnormal_signals=np.asarray(rows1)
  return raw_normal_signals,raw_Abnormal_signals

def signal_pre_processing(x,y):
  # in this function I use for loops to help me filtering the signals and delete the high frequnce noice
  # taking in mind the range of the signals and the length 
  # side_note: this process could be used in any code to help clean any ECG dataset 
  s1=[]
  s2=[]
  l=x.shape[0]
  ll=y.shape[0]
  for i in range(l):
    b=scipy.fft.fft (x[i])
    for i in range(188) :
      if b[i].real >=8:
        b[i]=0
      if b[i].real <=-8:
        b[i]=0
    b=scipy.fft.ifft(b)
    s1.append(b)
  for i in range(ll):
    B=scipy.fft.fft (y[i])
    for i in range(188) :
      if B[i].real >=8:
        B[i]=0
      if B[i].real <=-8:
        B[i]=0
    B=scipy.fft.ifft(B)
    s2.append(B)
  Normal_signals=np.asarray(s1)
  AbNormal_signals=np.asarray(s2)
  return Normal_signals,AbNormal_signals

def Do_dataset(x,y):
  # building dataset after cleaning all the signals and establish a dataframe as well as define a colume of labels.
  l=x.shape[0]
  ll=y.shape[0]
  number=[]
  signals=[]
  for i in range(l):
    signals.append(x[i])
    number.append(0)
  for i in range(ll):
    signals.append(y[i])
    number.append(1)
  numbers=np.asarray(number)
  # constructe dataset using data frame 
  my_series=pd.Series(data=signals,name='signals')
  data=pd.DataFrame(my_series)
  data['label']=numbers
  return data

def load_signal_label(x):
  signals = []
  label = []
  Signals_data=[]
  for i in range(len(x)):
    indexed_data = x.iloc[i]
    ff = indexed_data['signals']
    i = 0
    signals.append(indexed_data['signals'])
    label.append(indexed_data['label'])
  signals = np.asarray(signals)
  label   = np.asarray(label)
  return signals, label

def data_gen(x,y):
  batch_signals = []
  batch_label = []
  # steps 
  for i in range(len(x)):
    sig=x[i]
    sig = np.expand_dims(sig, axis=-1)
    batch_signals.append(sig)
    labell =y[i]
    batch_label.append(labell)
  A=np.asarray(batch_signals)
  B=np.asarray(batch_label)
  return A, B

def steps(x):
  Signals=[]
  for i in range(len(x)):
    X = x[i]
    X = np.expand_dims(X, axis=0)
    X = np.resize(X,[10,188])
    n=0
    Z=[]
    for j in range(10):
      step=n+10
      Y=X[j]
      for k in range(188):
        if k<=step and k>n:
          Y[k]=Y[k]
        else:
          Y[k]=0
      n=n+18
      Z.append(Y)
    Signals.append(Z)
  Signals=np.asarray(Signals)
  return Signals

file = open(r'/content/drive/MyDrive/Colab Notebooks/ECG Signal /ptbdb_normal.csv')
csvreader = csv.reader(file)
file1 = open(r'/content/drive/MyDrive/Colab Notebooks/ECG Signal /ptbdb_abnormal.csv')
csvreader1 = csv.reader(file1)
Normal,AbNormal= Import_data(csvreader,csvreader1)
normal,Abnormal= signal_pre_processing(Normal,AbNormal)
Dataset= Do_dataset(normal,Abnormal)
Signals, label= load_signal_label(Dataset)
Signals=steps(Signals)
xx_train_gen, yy_train_gen = data_gen(Signals, label)

def action_model(shape=(10,188, 1), nbout=2):
    
  def Convnet(shape=(188,1)):
    model= Sequential()
    model.add(Conv1D(32,3,input_shape=shape,strides=1,padding='same', activation='relu'))

    model.add(MaxPool1D(pool_size=2))

    model.add(Conv1D(64,5,strides=1,padding='same', activation='relu'))
    
    model.add(MaxPool1D(pool_size=2))

    model.add(Conv1D(128,7,input_shape=shape,strides=1,padding='same', activation='relu'))
  
    model.add(MaxPool1D(pool_size=2))

    model.add(Conv1D(256,9,input_shape=shape,strides=1,padding='same', activation='relu'))

    model.add(MaxPool1D(pool_size=2))

    model.add(GlobalMaxPool1D())
    return model

  model= Sequential()

  model.add(TimeDistributed(Convnet(shape[1:]),input_shape=shape))
  model.add(LSTM(64))
  model.add(Dense(1024, activation='sigmoid'))

  model.add(Dense(512, activation='sigmoid'))

  model.add(Dense(256, activation='sigmoid'))

  model.add(Dense(128, activation='sigmoid'))

  model.add(Dense(64, activation='sigmoid'))

  model.add(Dense(32, activation='sigmoid'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(nbout, activation='sigmoid'))
  return model

signal_dim = (188,1)
steps= (10) 
classes = 2
INPUT_SHAPE= (steps,)+signal_dim
print(INPUT_SHAPE)
model = action_model(INPUT_SHAPE, 2)
from sklearn.model_selection import train_test_split
# Split the data
x_train, x_valid, y_train, y_valid = train_test_split(xx_train_gen, yy_train_gen, test_size=0.33, shuffle= True)
model.compile(optimizer=tf.keras.optimizers.Adam(0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
history = model.fit(x=x_train, y= y_train,validation_data=(x_valid, y_valid), epochs =23, batch_size =32, verbose = 1, shuffle = 1)

