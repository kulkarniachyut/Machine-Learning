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

for j in arr :
    for n,i in enumerate(list(samples[:,j])):
        if i == -1:
            samples[:, j][n] = 0


a = (np.zeros(2000)).transpose()
for i in range(0,21):
    samples = np.c_[samples,a]
k=30
for i in change:
    for j in range(0,len(samples[:, i])):
        if  samples[:, i][j] == -1:
            samples[:, k][j] = 1
            samples[:, k+1][j] = 0
            samples[:, k+2][j] = 0
        if samples[:, i][j] == 0:
            samples[:, k][j] = 0
            samples[:, k + 1][j] = 1
            samples[:, k + 2][j] = 0
        else :
            samples[:, k][j] = 0
            samples[:, k + 1][j] = 0
            samples[:, k + 2][j] = 1
    k=k+3


lin = [];poly1 = [];poly2 = [];poly3 = [];rbf = []

def linear() :
    for i in range(-6,3):
        lin.append([svm_train((labels).tolist(),(samples).tolist(),  "-t 0 -c "  +str(4**i)+  " -v 3 -q ") , i]);

start_time = time.time()
linear();
print("--- %s seconds ---" % ((time.time() - start_time)/9.0))
bestC = []


def polynomial():
    for i in range(-3, 8):
        poly1.append([svm_train((labels).tolist(), (samples).tolist(), " -t 1 -d 1 -c " + str(4 ** i) + " -v 3 -q ") , i]);
        poly2.append([svm_train((labels).tolist(), (samples).tolist(), "  -t 1 -d 2 -c " + str(4 ** i) + " -v 3 -q "),i]);
        poly3.append([svm_train((labels).tolist(), (samples).tolist(), "  -t 1 -d 3 -c " + str(4 ** i) + " -v 3 -q "),i]);

start_time = time.time()
polynomial();
print("--- %s seconds ---" % ((time.time() - start_time)/27.0))


def kernel():
    for i in range(-3, 8):
        for j in range(-7, -1):
            bestC.append([svm_train((labels).tolist(), (samples).tolist(), " -t 2  -g "+str(4**j)+ " -c " + str(4 ** i) + " -v 3 -q "),i ,j ]);


start_time = time.time()
kernel();
print("--- %s seconds ---" % ((time.time() - start_time)/28.0))

#
print '  ..... RBF .......'
v =[]
for i in range(0,len(bestC)) :
    v.append(bestC[i][0])
print bestC[np.argmax(np.array(v))]
print v

# print '  ..... LINEAR  .......'
# v= []
# for i in range(0,len(lin)) :
#     v.append(lin[i][0])
# # print v
# print lin[np.argmax(np.array(v))]
#
# print '  ..... POLYNOMIAL 1 .......'
# v= []
# for i in range(0,len(poly1)) :
#     v.append(poly1[i][0])
# # print v
# print poly1[np.argmax(np.array(v))]
#
#
# print '  ..... POLYNOMIAL 2 .......'
# v= []
# for i in range(0,len(poly2)) :
#     v.append(poly2[i][0])
# # print v
# print poly2[np.argmax(np.array(v))]
#
#
# print '  ..... POLYNOMIAL 3 .......'
# v= []
# for i in range(0,len(poly3)) :
#     v.append(poly3[i][0])
# # print v
# print poly3[np.argmax(np.array(v))]
#
#
#
