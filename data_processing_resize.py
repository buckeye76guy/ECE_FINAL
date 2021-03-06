# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:05:44 2017

@author: Josiah Hounyo

This script will create the training and testing sets.
I wish to use 75% of the number of pictures in each folder
for training and 25% for testing
"""

from scipy.misc import imresize
import scipy.ndimage as nd
import numpy as np
import os

# current directory has two folders
flders = ['Cyrillic', 'Latin']

# get the number of observations to be placed in the train
# and test set
nrow_tr, nrow_te = 0, 0
for fld in flders:
    os.chdir(fld)
    # inside each directory is a list of directories
    subflders = os.listdir() # subfolders
    for subfld in subflders:
        nrow = len(os.listdir(subfld)) # number of photos in current subfolder
        add_tr = int(np.floor(75*nrow/100)) # to be added to nrow_tr
        nrow_tr += add_tr
        nrow_te += nrow - add_tr
    os.chdir('..') # get back to parent

x_train = np.zeros((nrow_tr, 28, 28)) # I already know size of images
# y_train = np.zeros((nrow_tr, 1))
y_train = [] # because they are characters! use int mapping for each :)
x_test = np.zeros((nrow_te, 28, 28))
# y_test = np.zeros((nrow_te, 1))
y_test = []

# it will be assumed that all photos have the same shape.
# instead of using gaussian filter, I will resize to 28 by 28
ind_tr, ind_te = 0, 0 # cursor to keep track of location in arrays
for fld in flders:
    os.chdir(fld)
    subflders = os.listdir()
    for subfld in subflders:
        nrow = len(os.listdir(subfld))
        add_tr = int(np.floor(75*nrow/100))
        # y_train/test is the name of the folder
        os.chdir(subfld)
        k = 0 # number of items to add to train set
        pics = os.listdir() # pictures in folder
        while k < len(pics):
            img = nd.imread(pics[k])
            if sorted(img.shape) != [4, 278, 278]:
                k += 1 # only this needs be increased
                # this way, ind_tr and ind_te will be such that only ending
                # rows in our matrices are all 0's. easily removed
                continue
            img = img.reshape((278, 278*4))
            img = imresize(img, (28,28))
            if k < add_tr:
                x_train[ind_tr,:,:] = img
                y_train.append(subfld) # subfld is a character
                ind_tr += 1
            else:
                x_test[ind_te,:,:] = img
                y_test.append(subfld)
                ind_te += 1
            k += 1
        os.chdir('..')
    os.chdir('..')

# process y_train and y_test: this will allow us to save them as integers
chars = list(set(y_train))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
y_train = list(map(lambda x: char_to_ix.get(x), y_train))
y_test = list(map(lambda x: char_to_ix.get(x), y_test))

np.save('xtrain.npy', x_train)
np.save('xtest.npy', x_test)
np.save('ytrain.npy', np.array(y_train))
np.save('ytest.npy', np.array(y_test))

# some characters in both Latin and Curillic are the same ...