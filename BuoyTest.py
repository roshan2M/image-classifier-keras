from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras.models import load_model

from PIL import Image
import numpy as np
from numpy import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2

model = load_model('buoy_classifier.h5')

buoy = cv2.imread("buoy_test.jpg")
buoy2 = cv2.imread("buoy_test_4.jpg")
img_rows, img_cols = 200, 200
buoy.resize((1, 3, img_rows, img_cols))
buoy2.resize((1, 3, img_rows, img_cols))
buoy_matrix = buoy.flatten()
buoy2_matrix = buoy.flatten()

fit = model.predict(buoy, batch_size=1, verbose=1)
print(fit)

fit2 = model.predict(buoy2, batch_size=1, verbose=1)
print(fit2)
