
# coding: utf-8

# Feature Extraction

#Jenny Zhang T05902136
import numpy as np
import sys
import csv
import pandas as pd
import scipy

training_data = open(sys.argv[1], 'r',encoding = "ISO-8859-1")
row = csv.reader(training_data, delimiter = ",")
data = []
for i in range(18):
	data.append([])

row_counter = 0;
for r in row:
	if row_counter != 0:
		for i in range(3,27):
			if r[i] != "NR":
				data[(row_counter-1)%18].append(float(r[i]))
			else:
				data[(row_counter-1)%18].append(float(0))
	row_counter = row_counter + 1
training_data.close()

train_x = []
train_y = []
hours = range(2,9)
#features: AMB_TEMP, NOx, O3, PM10, PM2.5, RH
features = [0,7,8,9,11,16]
squares = [8,9]
rows = len(features)

for i in range(12):
	for j in range(471):
		train_x.append([0.5])
		for f in features:
			for s in hours:
				train_x[471*i+j].append(data[f][480*i+j+s])
				if f in squares:
					train_x[471*i+j].append((data[f][480*i+j+s])**2)
		train_y.append(data[9][480*i+j+9])

n = len(train_y)
p = 1 + len(hours) * (rows + len(squares))
train_x = np.asarray(train_x)
train_y = np.asarray(train_y).reshape((len(train_y),1))


# Polynomial Regression and Gradient Descent
weight_vector = np.dot(np.dot(np.linalg.inv(np.dot(train_x.T, train_x)), train_x.T),train_y)
prev_gra = np.zeros([p,1], dtype = float)
learning_rate = 0.0001
iterations = 20000

for i in range(iterations):
    y_prime = np.dot(train_x, weight_vector)
    L = y_prime - train_y
    gra = 2*np.dot(train_x.T,L)
    prev_gra += gra**2
    ada = np.sqrt(prev_gra)
    weight_vector = weight_vector - learning_rate * gra/ada


# PM 2.5 Calculation
testing_data = open(sys.argv[2], "r")
row = csv.reader(testing_data, delimiter = ",")
test_x = []
    
rows_counter = -1
row_counter = 0
for r in row:
    if row_counter%18 == 0:
        rows_counter = rows_counter + 1
        test_x.append([0.5])
    for i in range(11-len(hours),11):
        if row_counter%18 in features:
            if r[i] != "NR":
                test_x[rows_counter].append(float(r[i]))
                if row_counter%18 in squares:
                    test_x[rows_counter].append(float(r[i])**2)
            else:
                test_x[rows_counter].append(float(0))
                if row_counter%18 in squares:
                    test_x[rows_counter].append(float(0))

    row_counter = row_counter + 1                                     
testing_data.close()

test_y = np.dot(test_x, weight_vector)


with open(sys.argv[3],'w') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    writer.writerow(["id","value"])
    row_counter = 0
    for row in test_y:
        writer.writerow(['id_'+str(row_counter),float(row[0])])
        row_counter = row_counter + 1
