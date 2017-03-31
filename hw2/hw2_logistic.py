
# coding: utf-8

# In[135]:

import numpy as np
import sys
import csv
import pandas as pd
import scipy
import math

train_x = np.genfromtxt(sys.argv[3],dtype = float, delimiter = ',',skip_header=1)
train_y = np.genfromtxt(sys.argv[4],dtype = float)
test_x = np.genfromtxt(sys.argv[5], dtype = float, delimiter = ',',skip_header=1)

#omits certain feature that negatively impacts our accuracy
# 1 = fnlwgt
omitted = [1]
sums = []
bias = np.ones([train_x.shape[0],1])
train_x = np.append(train_x, bias, 1)
for s in sums:
    train_x[:,s[0]] = np.sum(train_x[:,s], axis=1)
train_x = np.delete(train_x, omitted, 1)
train_y = train_y.reshape((len(train_y),1))

#sigmoid function
def sigmoid(X):
    return 1.0/(1.0+np.exp(-1.0*X))

validation_size = 0

TV_x = np.split(train_x, [train_x.shape[0]-validation_size])
train_x = TV_x[0]
validation_x = TV_x[1]
TV_y = np.split(train_y, [train_y.shape[0]-validation_size])
train_y = TV_y[0]
validation_y = TV_y[1]

#declare our parameters
weight_vector = np.empty([train_x.shape[1],1],dtype = float)
weight = 0.0
weight_vector.fill(weight)
learning_rate = 0.02
prev_gra = np.zeros([train_x.shape[1],1], dtype = float)
iterations = 4000
reg = 0.0001

def shuffle(X, Y):
    randomize = np.arrage(len(X))
    np.random.shuffle(randomize)
    return(X[randomize], Y[randomize])

#for i in range(epoch_num):
#    train_x, train_y = shuffle(train_x, train_y)
#    epoch_loss = 0.0
#    for idx in range(batch_num):
#        y = sigmoid(np.dot(train_x,weight_vector)+bias)
#        L = train_y - y
#        gra = np.dot(train_x.T, L)
#        b_gra = np.sum(L)
#        prev_gra += gra**2
#        ada = np.sqrt(prev_gra)
#        weight_vector = weight_vector + learning_rate *(gra/ada + reg*weight_vector)
#        bias = bias + learning_rate *(b_gra/ada)
  
#logistic regression & gradient descent      
for i in range(iterations):
    y = sigmoid(np.dot(train_x,weight_vector))
    L = train_y - y
    gra = np.dot(train_x.T,L)
    prev_gra += gra**2
    ada = np.sqrt(prev_gra)
    weight_vector = weight_vector + learning_rate *(gra/ada+reg*weight_vector)
    

counter = 0
matched = 0
prediction_y = sigmoid(np.dot(train_x, weight_vector))
for row in prediction_y:
    if row[0] >= 0.5:
        if bool(train_y[counter]):
            matched = matched + 1
    else:
        if not bool(train_y[counter]):
            matched = matched + 1
    counter = counter +1
    
accuracy = float(matched)/train_y.shape[0]
print(prediction_y[0:10])
print(accuracy*100)

#prediction
bias = np.ones([test_x.shape[0],1])
test_x = np.append(test_x, bias, 1)
for s in sums:
    test_x[:,s[0]] = np.sum(test_x[:,s], axis=1)
test_x = np.delete(test_x, omitted, 1)
test_y = sigmoid(np.dot(test_x, weight_vector))


#write to our file
with open(sys.argv[6],'w') as csvfile:
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

