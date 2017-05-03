import numpy as np
import sys
import csv
import scipy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2s'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import MaxPooling2D, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model

model = load_model("myModel.h5")

def load_data():
    train_data =open(sys.argv[1], 'r')
    row = csv.reader(train_data, delimiter = ",")
    test_x = []
    test_y = []

    row_counter = 0
    for r in row:
        if row_counter is not 0:
            test_y.append(r[0])
            test_x.append(np.reshape(r[1].split(),(48,48)))
        row_counter += 1

    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)
    
    test_x = test_x.astype('float32')
    test_y = test_y.astype('int32')
    
    
    test_x = test_x/255
    test_y = np_utils.to_categorical(test_y)
    
    return (test_x, test_y)
    
(test_x, test_y)= load_data()

test_x=test_x.reshape(test_x.shape[0],48,48,1)


predict_y = model.predict(test_x, batch_size = 48,verbose = 0)

with open(sys.argv[2],'w') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    writer.writerow(["id","label"])
    row_counter = 0
    for row in predict_y:
        writer.writerow([row_counter,np.argmax(row)])
        row_counter = row_counter + 1
