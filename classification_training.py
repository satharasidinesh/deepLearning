# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:17:36 2018

@author: Dinesh Satharasi
"""
import numpy as np
import os,glob
import random
from scipy import misc
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import cv2
from sklearn.model_selection import train_test_split


IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
 
imageCount = sum([len(files) for r, d, files in os.walk("data")])
def getImageData():
    X = np.zeros((imageCount, IMAGE_WIDTH, IMAGE_WIDTH, 3), dtype='float64')
    Y = np.zeros(imageCount)
    count_data = 0
    for category1 in glob.glob("data\category1\*.jpg"):
        full_size_image = cv2.imread(category1)
        X[count_data] = cv2.resize(full_size_image, (IMAGE_WIDTH,IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
        Y[count_data] = 0
        count_data  = count_data+1;
    for category2 in glob.glob("data\category2\*.jpg"):
        full_size_image = cv2.imread(category2)
        X[count_data] = cv2.resize(full_size_image, (IMAGE_WIDTH,IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
        Y[count_data] = 1
        count_data  = count_data+1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X_train, X_test, Y_train, Y_test



X_train, X_test, Y_train, Y_test = getImageData()
Y_train = np_utils.to_categorical(Y_train, 2)
Y_test = np_utils.to_categorical(Y_test, 2)

X_train /= 255
X_test /= 255


class LeNet:
    def build(classes):
        model = Sequential()
        model.add(Conv2D(64, (4, 4), input_shape=(IMAGE_WIDTH, IMAGE_WIDTH, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (4, 4)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (4, 4 )))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        model.compile(loss='binary_crossentropy', optimizer=Adam() , metrics=['accuracy'])
        return model    
model = LeNet.build(classes=2)                  
history = model.fit(X_train, Y_train,
batch_size=20, epochs=25,
verbose=1, validation_split=0.1)
model.save("model.h5")
#print(len(image_data))