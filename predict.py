# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:20:25 2018

@author: Dinesh Satharasi
"""

import cv2
import numpy as np
from keras.models import load_model

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
full_size_image = cv2.imread("c2.jpg")
X = np.zeros((1, IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype='float64')
X[0] = cv2.resize(full_size_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
model = load_model('model.h5')
print(model.predict(X))

