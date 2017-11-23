from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils

import numpy as np
import pygu
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

path1 = 'D:\GitHub\Side Projects\Image Classification\Test_images'
path2 = 'D:\GitHub\Side Projects\Image Classification\Test_images_resized'

listing = os.listdir(path1)
num_buoys = size(listing)

img_rows, img_cols = 200, 200

print num_buoys

for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((img_rows, img_cols))
    colour = img.convert('RGB')
    colour.save(path2 + '\\' + file, 'JPEG')

imlist = os.listdir(path2)

im1 = array(Image.open('Test_images_resized' + '\\' + imlist[0]))
m, n = im1.shape[0:2]  # get the size of images
imnum = len(imlist)  # number of images

# Create matrix to store flattened images
image_matrix = array([array(Image.open('Test_images_resized' + '\\' + im2)).flatten()
                      for im2 in imlist], 'f')

label = np.ones((num_buoys,), dtype=int)
label[:20] = 0

data, dataLabel = shuffle(image_matrix, label, random_state=2)
train_data = [data, dataLabel]

print(train_data[0].shape)
print(train_data[1].shape)

batch_size = 15
nb_classes = 2  # number of classes to train on
nb_epoch = 10  # number of epochs (intervals) to test
nb_filters = 32  # number of filters
nb_pool = 2  # number of pixels for the MaxPooling2D layer
nb_conv = 3  # layers of cnn applied to the model

(X, y) = (train_data[0], train_data[1])

# Split X and y into training + testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=4)

print ("X_train.shape: " + str(X_train.shape))
print ("X_test.shape: " + str(X_test.shape))

X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Design the model architecture

# Sequential model is a stack of layers/filters that are applied to
# the data
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid',
                        input_shape=(3, img_rows, img_cols)))

# An activation is an assignment of weights to neurons in that layer
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta',
              metrics=['accuracy'])

# Fit the model to the data and test accuracy
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])

# Save the model for testing
model.save('buoy_classifier.h5')
