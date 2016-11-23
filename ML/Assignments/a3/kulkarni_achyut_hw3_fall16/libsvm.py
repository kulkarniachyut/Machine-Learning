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

Gamma = 4**(-3)
C = 4**(3)


m = svm_train((labels).tolist(),(samples).tolist(),  "-s 0 -t 2 -g 0 -c " + str(C)+ " -g " + str(Gamma) +"  ")
print svm_predict((testlabels).tolist(),(testsamples).tolist(),m)