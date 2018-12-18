# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:25:52 2018

@author: Xiaomi
"""

import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM


pos = pd.read_csv('positive_set.csv',sep=';',index_col='index')
neg = pd.read_csv('negative_set.csv',sep=';',index_col='index')

pos = pos + 1
neg = neg + 1 
pos = pos.fillna(0)
neg = neg.fillna(0)
#
pos = sequence.pad_sequences(pos.values, maxlen=40)
neg = sequence.pad_sequences(neg.values, maxlen=40)

pos = np.c_[pos,np.ones(len(pos))]
neg = np.c_[neg,np.zeros(len(neg))]
alls = np.r_[pos,neg]

randomize = np.arange(len(alls))
np.random.shuffle(randomize)
alls = alls[randomize]

x_train = alls[:28000]
y_train = x_train[:,40]
x_train = np.delete(x_train,40,axis=1)

x_test = alls[28000:30000]
y_test = x_test[:,40]
x_test = np.delete(x_test,40,axis=1)

max_features = 200000

batch_size = 64

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))