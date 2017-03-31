
# coding: utf-8


import numpy as np
import sys
import csv
import pandas as pd
import scipy
import math

train_x = np.genfromtxt(sys.argv[13,dtype = float, delimiter = ',',skip_header=1)
train_y = np.genfromtxt(sys.argv[4],dtype = float)

omitted = [1]
sums = []
for s in sums:
    train_x[:,s[0]] = np.sum(train_x[:,s], axis=1)
train_x = np.delete(train_x, omitted, 1)

test_x = np.genfromtxt(sys.argv[5], dtype = float, delimiter = ',',skip_header=1)
for s in sums:
    test_x[:,s[0]] = np.sum(test_x[:,s], axis=1)
test_x = np.delete(test_x, omitted, 1)


validation_size = 0

counter = 0
class0_size = 0
class1_size = 0

dim = train_x.shape[1]
class0 = np.zeros((dim,))
class1 = np.zeros((dim,))
class0_cov = np.zeros((dim, dim))
class1_cov = np.zeros((dim, dim))

for i in range(train_x.shape[0]):
    if bool(train_y[i]):
        class1_size += 1
        class1 += train_x[i]
    else:
        class0_size += 1
        class0 += train_x[i]
    
class0_mean = class0 / class0_size
class1_mean = class1 / class1_size

for i in range(train_x.shape[0]):
    if bool(train_y[i]):
        class1_cov += np.dot(np.transpose([train_x[i] - class1_mean]), ([train_x[i] - class1_mean]))
    else:
        class0_cov += np.dot(np.transpose([train_x[i] - class0_mean]), ([train_x[i] - class0_mean]))
        
class0_cov = class0_cov / class0_size
class1_cov = class1_cov / class1_size

total_cov = (class0_size/total_size)*class0_cov + (class1_size/total_size)*class1_cov


#sigmoid function
def sigmoid(z):
    ans = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999) 

#probability function
def probablity(x, mean0, mean1, cov):
    cov_inv = np.linalg.inv(cov)
    w = np.dot((mean0-mean1), cov_inv)
    x = x.T
    b = -0.5*np.dot(np.dot((mean0),cov_inv),mean0)+0.5*np.dot(np.dot((mean1),cov_inv),mean1)+math.log(float(class0_size)/class1_size)
    z = np.dot(w,x) + b
    return 1.0/(1.0+math.exp(-z))



prob_y = []
counter = 0
for row in train_x:
    prob_y.append(probablity(row, class0_mean, class1_mean, total_cov))
    counter = counter + 1


matched = 0
counter = 0
 
for i in range(len(prob_y)):
    if prob_y[i] < 0.5 and bool(train_y[i]):
        matched = matched + 1
    elif prob_y[i] >= 0.5 and not bool(train_y[i]):
        matched = matched + 1

accuracy = float(matched)/train_y.shape[0]
print(accuracy)


#predict
test_y = []
for row in test_x:
    test_y.append(probablity(row, class0_mean, class1_mean, total_cov))



#write prediction to file
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

