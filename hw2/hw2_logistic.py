
# coding: utf-8

# In[8]:

import numpy as np
import sys
import csv
import pandas as pd
import scipy
import math

train_x = np.genfromtxt(sys.argv[3],dtype = float, delimiter = ',',skip_header=1)
train_y = np.genfromtxt(sys.argv[4],dtype = float)
test_x = np.genfromtxt(sys.argv[5], dtype = float, delimiter = ',',skip_header=1)

omitted = [1,11,12,13,14,16,17,18,19,20,21,42,43,44,51,52,105] 
sums = [[15,16,17,18,19,20,21],[7,12],[8,13],[9,14],[10,11],[40,42,43,44,51],[102,105]]
for s in sums:
    train_x[:,s[0]] = np.sum(train_x[:,s], axis=1)
    test_x[:,s[0]] = np.sum(test_x[:,s], axis=1)
test_x = np.delete(test_x, omitted, 1)
train_x = np.delete(train_x, omitted, 1)


# In[9]:

def sigmoid(X):
    ans = 1.0/(1.0+np.exp(-1.0*X))
    return np.clip(ans, 0.00000000000001, 0.99999999999999)

def feature_normalize(X_train, X_test):
    # feature normalization with all X
    X_all = np.concatenate((X_train, X_test))
    mu = np.mean(X_all, axis=0)
    sigma = np.std(X_all, axis=0)
    
    # only apply normalization on continuos attribute
    index = [0, 2, 3, 4]
    mean_vec = np.zeros(X_all.shape[1])
    std_vec = np.ones(X_all.shape[1])
    mean_vec[index] = mu[index]
    std_vec[index] = sigma[index]

    X_all_normed = (X_all - mean_vec) / std_vec

    # split train, test again
    X_train_normed = X_all_normed[0:X_train.shape[0]-1]
    X_test_normed = X_all_normed[X_train.shape[0]:]
    return X_train_normed, X_test_normed

train_x, test_x = feature_normalize(train_x, test_x)

validation_size = 8000

TV_x = np.split(train_x, [train_x.shape[0]-validation_size])
train_x = TV_x[0]
validation_x = TV_x[1]
TV_y = np.split(train_y, [train_y.shape[0]-validation_size])
train_y = TV_y[0]
validation_y = TV_y[1]

train_size = train_x.shape[0]
weight_vector = np.zeros([train_x.shape[1],],dtype = float)
bias = np.zeros([1,],dtype = float)
learning_rate = 0.015
prev_gra = np.zeros([train_x.shape[1],], dtype = float)
prev_gra_b = np.zeros([1,],dtype = float)
prev_gra.fill(0.00000001)
prev_gra_b.fill(0.00000001)
reg = 0.0001
epoch_num = 1000
batch_size = 64
batch_num = int(math.floor(train_size / batch_size))

def shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return(X[randomize], Y[randomize])
    
for epoch in range(epoch_num):
        train_x, train_y = shuffle(train_x, train_y)
        epoch_loss = 0.0
        for i in range(batch_num):
            X = train_x[i*batch_size:(i+1)*batch_size]
            Y = train_y[i*batch_size:(i+1)*batch_size]
            y = np.asarray(sigmoid(np.dot(X,weight_vector)+bias))
            L = Y - y
            gra = np.dot(X.T,L)
            prev_gra += gra**2
            prev_gra_b = np.sum(L)**2
            ada = np.sqrt(prev_gra)
            ada_b = np.sqrt(prev_gra_b)
            weight_vector = weight_vector + learning_rate *(gra/ada+reg*weight_vector)
            bias = bias + learning_rate*np.sum(L/ada_b)
            cross_entropy = -(np.dot(Y, np.log(y)) + np.dot((1 - Y), np.log(1 - y)))
            epoch_loss += cross_entropy



test_y = sigmoid(np.dot(test_x, weight_vector)+bias)
with open(sys.argv[6],'w') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    writer.writerow(["id","label"])
    row_counter = 1
    for row in test_y:
        if row >=0.5:
            boolean = int(1)
        else:
            boolean = int(0)
        writer.writerow([row_counter,boolean])
        row_counter = row_counter + 1

