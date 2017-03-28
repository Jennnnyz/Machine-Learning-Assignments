import numpy as np
import sys
import csv
import pandas as pd
import scipy
import math

train_x = np.genfromtxt("X_train.csv",dtype = float, delimiter = ',',skip_header=1)
train_y = np.genfromtxt("Y_train.csv",dtype = float)

omitted = [1]
bias = np.ones([train_x.shape[0],1])
train_x = np.append(train_x, bias, 1)
train_x = np.delete(train_x, omitted, 1)
train_y = train_y.reshape((len(train_y),1))

def sigmoid(X):
    return 1.0/(1.0+np.exp(-1.0*X))

validation_size = 8000

TV_x = np.split(train_x, [train_x.shape[0]-validation_size])
train_x = TV_x[0]
validation_x = TV_x[1]
TV_y = np.split(train_y, [train_y.shape[0]-validation_size])
train_y = TV_y[0]
validation_y = TV_y[1]


p = train_x.shape[1]
weight_vector = np.empty([p,1],dtype = float)
weight = 0.0
weight_vector.fill(weight)
learning_rate = 0.015
prev_gra = np.zeros([p,1], dtype = float)
iterations = 3000
reg = 1

for i in range(iterations):
    y = sigmoid(np.dot(train_x,weight_vector))
    L = y - train_y
    gra = np.dot(train_x.T, L)
    prev_gra += gra**2
    ada = np.sqrt(prev_gra)
    weight_vector = weight_vector - learning_rate *(gra/ada+reg*weight_vector)

counter = 0
matched = 0
prediction_y = sigmoid(np.dot(validation_x, weight_vector))
for row in prediction_y:
    if row[0] >= 0.5:
        if bool(validation_y[counter]):
            matched = matched + 1
    else:
        if not bool(validation_y[counter]):
            matched = matched + 1
    counter = counter +1
    
accuracy = float(matched)/validation_y.shape[0]
print(prediction_y[0:10])
print(accuracy)

test_x = np.genfromtxt("X_test.csv", dtype = float, delimiter = ',',skip_header=1)
bias = np.ones([test_x.shape[0],1])
test_x = np.append(test_x, bias, 1)
test_x = np.delete(test_x, omitted, 1)
test_y = sigmoid(np.dot(test_x, weight_vector))

with open('prediction.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    writer.writerow(["id","label"])
    row_counter = 1
    for row in test_y:
        if row[0] >=0.5:
            boolean = int(1)
        else:
            boolean = int(0)
        writer.writerow([row_counter,boolean])
        row_counter = row_counter + 1