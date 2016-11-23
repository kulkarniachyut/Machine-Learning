import scipy.io as sio
import numpy as np
from svmutil import *
import time

trainMat = sio.loadmat('phishing-train.mat');
testMat = sio.loadmat('phishing-test.mat');
samples = np.array(trainMat['features'])
labels = np.array(trainMat['label'][0])
testlabels = np.array(testMat['label'][0])
testsamples = np.array(testMat['features'])

arr = [0,2,3,4,5,8,9,10,11,12,15,16,17,18,19,20,21,22,23,24,26,27,29]
change = [1,6,7,13,14,25,28]

def features(samples2):
    for j in arr:
        for n, i in enumerate(list(samples2[:, j])):
            if i == -1:
                samples2[:, j][n] = 0

    a = (np.zeros(2000)).transpose()
    for i in range(0, 21):
        samples2 = np.c_[samples2, a]
    k = 30
    for i in change:
        for j in range(0, len(samples2[:, i])):
            if samples2[:, i][j] == -1:
                samples2[:, k][j] = 1
                samples2[:, k + 1][j] = 0
                samples2[:, k + 2][j] = 0
            if samples2[:, i][j] == 0:
                samples2[:, k][j] = 0
                samples2[:, k + 1][j] = 1
                samples2[:, k + 2][j] = 0
            else:
                samples2[:, k][j] = 0
                samples2[:, k + 1][j] = 0
                samples2[:, k + 2][j] = 1

        k = k + 3
    return samples2

samples1 = features(samples)
Gamma = 4**(-1)
C = 4**(8)
m = svm_train((labels).tolist(),(samples1).tolist(),  "-s 0 -t 2 -c " + str(C)+ " -h 0 -g  " + str(Gamma) +"  ")
# print m
samples1 = features(testsamples)
print svm_predict((testlabels).tolist(),(testsamples).tolist(),m)