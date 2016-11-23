import numpy as np
from collections import Counter
from scipy.spatial import distance
import math as m
import pandas as pd

# trainMatrix = np.loadtxt(open("/Users/achi/Desktop/train.txt","rb"),delimiter=",")
trainMatrix = np.loadtxt(open("./train.txt","rb"),delimiter=",")
testMatrix = np.loadtxt(open("./test.txt","rb"),delimiter=",")

trainData = pd.read_csv("./train.csv" ,header=None)
trainingD =  pd.DataFrame(trainData[trainData.columns[1:11]])

result = []

mean = (pd.DataFrame(trainData[trainData.columns[1:10]])).mean()
var = (pd.DataFrame(trainData[trainData.columns[1:10]])).var()


def euclidean(a,b) :
    dist=0.0
    for i in range(0,len(a)) :
        dist+= (a[i] - b[i])**2
    return m.sqrt(dist)

def manhattan(a,b) :
    dist=0.0
    for i in range(0,len(a)) :
        dist+= m.fabs(a[i]-b[i])
    return dist

def normalise(a,b) :
    for i in range(0,len(a)) :
        if( var[i+1] == 0.0) :
            a[i]=((a[i]-mean[i+1]))
            b[i] = ((b[i] - mean[i + 1]))
        else :
            a[i] = ((a[i] - mean[i + 1]) / (m.sqrt(var[i + 1])))
            b[i] = ((b[i] - mean[i + 1])/(m.sqrt(var[i + 1])))
    # print a
    # print b
    return [a,b]

def testingAccuracy(K,type):
    true = 0;correct=0;
    false = 0;wrong=0;
    for testrow in testMatrix :
        result=[] ; minval = [];minval1 = [];count=0;result1 = [];minval1=[];count1=0
        for trainrow in trainMatrix :
            a = np.array(trainrow[1:len(trainrow)-1])
            b=  np.array(testrow[1:len(testrow)-1])
            vectors = normalise(a,b)
            Edist = euclidean(vectors[0], vectors[1])
            Mdist = manhattan(vectors[0], vectors[1])
            result.append([Edist , trainrow[len(trainrow)-1]])
            result1.append([Mdist, trainrow[len(trainrow) - 1]])
        # print result
        result.sort(key=lambda row: row[0:])
        result1.sort(key=lambda row: row[0:])

        # print "sorted" , result
        for k in range(0,K) :
            minval.append(result[k][1])
            minval1.append(result[k][1])
        # print "ans = " , minval
        count = Counter(minval)
        count1 = Counter(minval1)
        # print count.most_common()
        
        # print "actual = ", testrow[len(testrow) - 1]
        if (count.most_common()[0][0] == testrow[len(testrow) - 1]):
            true+= 1.0;
        else:
            false+= 1.0
        if (count1.most_common()[0][0] == testrow[len(testrow) - 1]):
            correct+= 1.0;
        else:
            wrong+= 1.0

    if(type == 1) :
        print 'accuracy for K = ', K, "=", (correct / (correct + wrong)) * 100, "%"
    else :
        print 'accuracy for K = ', K, "=", (true / (true + false)) * 100, "%"
    return


def trainingAccuracy(K, type) :
    minVal = [];
    minVal1 = [];
    true = 0;
    false = 0;
    correct = 0;
    wrong = 0;
    for i in range(0, len(trainMatrix)):
        result3=[];result4=[]
        initial = np.array(trainMatrix[i])
        a = initial[1:len(initial) - 1]
        b = np.delete(trainMatrix, (i), axis=0)
        for j in range(0, len(b)):
            c = np.array(b[j][1:len(b[j]) - 1])
            # vectors = normalise(a,b)
            Edist = euclidean(a, c)
            # Edist = euclidean(vectors[0], vectors[1])
            # Mdist = manhattan(vectors[0], vectors[1])
            Mdist = manhattan(a, c)
            result3.append([Edist, b[j][len(b[j]) - 1]])
            result4.append([Mdist, b[j][len(b[j]) - 1]])
            # print result3
            # print result4
        result3.sort(key=lambda row: row[0:])
        result4.sort(key=lambda row: row[0:])
        # print result3
        # print result4
        for k in range(0, K):
            minVal.append(result3[k][1])
            minVal1.append(result4[k][1])
        # print "top k elements" , minVal
        count = Counter(minVal);
        count1 = Counter(minVal1)
        if (count.most_common()[0][0] == initial[len(initial) - 1]):
            true += 1.0;
        else:
            false += 1.0
        if (count1.most_common()[0][0] == initial[len(initial) - 1]):
            correct += 1.0;
        else:
            wrong += 1.0;
        minVal = []
        minVal1 = []
    if (type == 2):
        print 'accuracy for K = ', K, "=", (true / (true + false)) * 100, "%"
    else:
        print 'accuracy for K = ', K, "=", (correct / (correct + wrong)) * 100, "%"


print "Testing Accuracy ----- >>>>"

print "Manhattan Distance Accuracy ----->>>"
testingAccuracy(1,1)
testingAccuracy(3,1)
testingAccuracy(5,1)
testingAccuracy(7,1)

print "Euclidean Distance Accuracy ------>>>"
testingAccuracy(1,2)
testingAccuracy(3,2)
testingAccuracy(5,2)
testingAccuracy(7,2)


print " Training Accuracy with LOO ---->>>"

print "Manhattan Distance Accuracy ----->>"
trainingAccuracy(1,1)
trainingAccuracy(3,1)
trainingAccuracy(5,1)
trainingAccuracy(7,1)

print "Euclidean Distance Accuracy ----->>"
trainingAccuracy(1,2)
trainingAccuracy(3,2)
trainingAccuracy(5,2)
trainingAccuracy(7,2)
