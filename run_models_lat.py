# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:37:08 2017

@author: Josiah Hounyo
"""

# may shuffle test set
from __future__ import print_function

import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')
from theano.tensor.shared_randomstreams import RandomStreams
from ZRNN import LSTM3, LSTM5

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_epoch_end(self, batch, logs={}):
        #print(logs)
        self.losses.append(logs.get('val_loss')) # losses
        self.accuracy.append(logs.get('val_acc')) # accuracy

batch_size = 32
nb_epochs = 100 # perhaps 50 epochs is enough
hidden_units = 100

X_train = np.load('xtrain_lat.npy')
max_pix = np.max(X_train)
X_train /= max_pix

X_test = np.load('xtest_lat.npy')
X_test /= max_pix # pretty sure maximum is same accross datasets

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = np.load('ytrain_lat.npy')
y_test = np.load('ytest_lat.npy')

nb_classes = np.unique(y_train).shape[0]
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

np.random.seed(3)
srgn = RandomStreams(3)

act = 'tanh'
lstm = LSTM3
eta_1 = 5e-4

lstm3 = lstm(consume_less='mem',output_dim=hidden_units,
                    activation=act,
                    input_shape=X_train.shape[1:])
lstm = LSTM5
lstm5 = lstm(consume_less='mem',output_dim=hidden_units,
                    activation=act,
                    input_shape=X_train.shape[1:])
def baseline_model(eta):
    model = Sequential()
    model.add(LSTM(hidden_units, activation=act, 
                   input_shape = X_train.shape[1:]))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    rmsprop = RMSprop(lr=eta, rho=0.9, epsilon=1e-8, decay=0)
    model.compile(loss='categorical_crossentropy',
              optimizer= 'rmsprop',
              metrics=['accuracy'])
    return model

def compare_model(eta, modl):
    model = Sequential()
    model.add(modl)
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    rmsprop = RMSprop(lr=eta, rho=0.9, epsilon=1e-8, decay=0)
    model.compile(loss='categorical_crossentropy',
              optimizer= 'rmsprop',
              metrics=['accuracy'])
    return model

# baseline model with different learning rates
history = LossHistory()
########################################################################
model = baseline_model(eta_1)
hist = model.fit(X_train, Y_train, batch_size=batch_size, 
                 nb_epoch=nb_epochs, verbose=1, 
                 validation_data=(X_test, Y_test), callbacks=[history])
my_array = np.hstack((np.matrix(history.losses).T,
                      np.matrix(history.accuracy).T))
np.save('latResults/base_eta1_d.npy', my_array) # d i.e. not from qsub command

eta = 1e-4
model = baseline_model(eta)
hist = model.fit(X_train, Y_train, batch_size=batch_size, 
                 nb_epoch=nb_epochs, verbose=1, 
                 validation_data=(X_test, Y_test), callbacks=[history])
my_array = np.hstack((np.matrix(history.losses).T,
                      np.matrix(history.accuracy).T))
np.save('latResults/base_eta2_d.npy', my_array)

########################################################################
# LSTM3 with different learning rates
model = compare_model(eta_1, lstm3)
hist = model.fit(X_train, Y_train, batch_size=batch_size,
                 nb_epoch=nb_epochs, verbose=1, 
                 validation_data=(X_test, Y_test), callbacks=[history])
my_array = np.hstack((np.matrix(history.losses).T,
                      np.matrix(history.accuracy).T))
np.save('latResults/lstm3_eta1_d.npy', my_array)

model = compare_model(eta, lstm3)
hist = model.fit(X_train, Y_train, batch_size=batch_size,
                 nb_epoch=nb_epochs, verbose=1, 
                 validation_data=(X_test, Y_test), callbacks=[history])
my_array = np.hstack((np.matrix(history.losses).T,
                      np.matrix(history.accuracy).T))
np.save('latResults/lstm3_eta2_d.npy', my_array)

########################################################################
# LSTM5 with different learning rates
model = compare_model(eta_1, lstm5)
hist = model.fit(X_train, Y_train, batch_size=batch_size,
                 nb_epoch=nb_epochs, verbose=1, 
                 validation_data=(X_test, Y_test), callbacks=[history])
my_array = np.hstack((np.matrix(history.losses).T,
                      np.matrix(history.accuracy).T))
np.save('latResults/lstm5_eta1_d.npy', my_array)

model = compare_model(eta, lstm5)
hist = model.fit(X_train, Y_train, batch_size=batch_size,
                 nb_epoch=nb_epochs, verbose=1, 
                 validation_data=(X_test, Y_test), callbacks=[history])
my_array = np.hstack((np.matrix(history.losses).T,
                      np.matrix(history.accuracy).T))
np.save('latResults/lstm5_eta2_d.npy', my_array)