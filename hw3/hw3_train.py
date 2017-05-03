
# coding: utf-8

# In[5]:

import numpy as np
import sys
import csv
import scipy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2s'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import MaxPooling2D, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2

np.random.seed(1414)

epochs = 10
batch_size = 80
pool_size1 = (2,2)
nb_filter1 = (3,3)
num_classes = 7

#sort our data based on train_y values
def sort_data(train_x, train_y):
    inds = train_y.argsort()
    train_x = train_x[inds]
    train_y = train_y[inds]
    return (train_x, train_y)

#dealing with unbalanced data
#     emotions: [0,1,2,3,4,5,6]
## of samples : [3995,436,4097,7215,4830,3171,4965]
def balance_data(train_x, train_y):

    #over sample emotion 1
    ids1 = np.where(train_y == 1)[0]
    train_x1 = train_x[ids1]
    train_y1 = train_y[ids1]
    for i in range(10):
        train_x = np.append(train_x,train_x1,axis = 0)
        train_y = np.append(train_y,train_y1, axis = 0)

    for i in [0,2,4,5,6]:
        ids = np.where(train_y == i)[0]
        x = train_x[ids]
        y = train_y[ids]
        train_x = np.append(train_x, x[:1500], axis = 0)
        train_y = np.append(train_y, y[:1500], axis = 0)

    return (train_x, train_y)

def validation_data(train_x, train_y):
    size = 700
    indexes = list(range(size))
    validation_x = train_x[indexes]
    validation_y = train_y[indexes]
    for i in range(6):
        i = i + 1
        ids = np.where(train_y == i)[0][:size]
        indexes.extend(ids)
        validation_x = np.append(validation_x, train_x[ids], axis = 0)
        validation_y = np.append(validation_y, train_y[ids], axis = 0)

    return (train_x, train_y),(validation_x, validation_y)

def shuffle_data(x, y):
    p = np.random.permutation(len(x))
    return (x[p], y[p])

def load_data():
    train_data =open(sys.argv[1], 'r')
    row = csv.reader(train_data, delimiter = ",")
    train_x = np.empty((28709,48,48),dtype ='float32')
    train_y = np.empty((28709),dtype='int32')

    row_counter = 0
    for r in row:
        if row_counter is not 0:
            train_y[row_counter-1] = r[0]
            train_x[row_counter-1] = (np.reshape(r[1].split(),(48,48)))
        row_counter += 1

    
    train_x = train_x.astype('float32')
    train_y = train_y.astype('int32')
    
    
    (train_x, train_y) = balance_data(train_x, train_y)
    (train_x, train_y) = sort_data(train_x, train_y)
    (train_x, train_y),(validation_x, validation_y) = validation_data(train_x, train_y)
    (validation_x, validation_y) = shuffle_data(validation_x, validation_y)
    (train_x, train_y) = shuffle_data(train_x, train_y)
    
    train_x = train_x/255.0
    train_y = np_utils.to_categorical(train_y)
    validation_y = np_utils.to_categorical(validation_y)
    
    return (train_x, train_y),(validation_x, validation_y)
    
(train_x,train_y),(validation_x,validation_y) = load_data()

#datagen = ImageDataGenerator(
#        rotation_range=10,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        horizontal_flip=True)


model = Sequential()
model.add(Conv2D(32,nb_filter1,input_shape=(48,48,1)))

model.add(Conv2D(32,nb_filter1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size1))
model.add(Dropout(0.25))

model.add(Conv2D(64,nb_filter1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size1))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1028))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
train_x=train_x.reshape(train_x.shape[0],48,48,1)
validation_x = validation_x.reshape(validation_x.shape[0], 48, 48, 1)
#datagen.fit(train_x)
callback = EarlyStopping(monitor = 'val_loss', min_delta = 0.00, patience = 1)
#model.fit_generator(datagen.flow(train_x, train_y, batch_size = batch_size), steps_per_epoch= train_x.shape[0]/batch_size, callbacks = [callback], validation_data = (validation_x, validation_y))
model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs, shuffle = True, callbacks = [callback], validation_data = (validation_x, validation_y))
score = model.evaluate(train_x, train_y , batch_size = batch_size)

model.save_weights("myWeights.h5")
