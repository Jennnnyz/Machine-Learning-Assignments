import numpy as np
import os
import sys
import csv
from math import log
from sklearn.svm import LinearSVR as SVR
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from PIL import Image

np.random.seed(17)


def load_data(testfile, trainfile, trainfile1):
    test_data = np.load(testfile)
    train_data = np.load(trainfile)
    train_data1 = np.load(trainfile1)
    return test_data, train_data, train_data1

#def load_images(folder):
#    data = []
#    for i in range(481):
#        name = folder + 'hand.seq' + str(i+1) + '.png'
#        img = Image.open(name)
#        img = img.crop((0,0,480,480))
#        img = img.resize((10,10), Image.ANTIALIAS)
#        img = np.array(img)
#        data.append(img)
#    data = np.asarray(data).reshape(481, 10*10)

def get_eigenvalues(data):
    SAMPLE = 20 # sample some points to estimate
    NEIGHBOR = 400 # pick some neighbor to compute the eigenvalues
    randidx = np.random.permutation(data.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(n_neighbors=NEIGHBOR,
                             algorithm='ball_tree').fit(data)

    sing_vals = []
    for idx in randidx:
        dist, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0,1:]]
        u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0))
        s /= s.max()
        sing_vals.append(s)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals

def save_results(filename, results):
    with open(filename,'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(["SetId","LogDim"])
        row_counter = 0
        for i in range(200):
            writer.writerow([i,results[i]])

def train(train_data, train_data1):
    X = train_data['X']
    y = train_data['y']

    X1 = train_data1['X']
    y1 = train_data1['y']

    X = np.concatenate((X, X1), axis = 0)
    y = np.concatenate((y, y1), axis = 0)

    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(X,y)
    return clf

def main():
    test_data, train_data, train_data1 = load_data(sys.argv[1], "large_data.npz", "large_data1.npz")

    clf = train(train_data, train_data1)

    test_X = []
    for i in range(200):
        data = test_data[str(i)]
        vs = get_eigenvalues(data)
        test_X.append(vs)
        #print(i)
    test_X = np.array(test_X)
    pred_y = clf.predict(test_X)

    save_results(sys.argv[2], pred_y)
    return 0
    


if __name__ == "__main__":
    main()