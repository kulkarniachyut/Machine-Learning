import sys
import os
import numpy as np
from collections import Counter
from scipy.spatial import distance


trainMatrix = np.loadtxt(open("/Users/achi/Desktop/train.txt","rb"),delimiter=",")
testMatrix = np.loadtxt(open("/Users/achi/Desktop/test.txt","rb"),delimiter=",")
result = []
result1 = []
result3 =[]
result4 =[]
testMatrix1 = [[68,1.52152,13.05,3.65,0.87,72.32,0.19,9.85,0.00,0.17,1],[213,1.51651,14.38,0.00,1.94,73.61,0.00,8.48,1.57,0.00,7]]

def testingAccuracy(K) :
    minVal = [] ; minVal1 = []; i=0;j=0; true=0;false=0;correct=0;wrong=0;
    for testrow in testMatrix :
    # for testrow in testMatrix1:
        j+=1;i=0;
        for trainrow in trainMatrix :
            a = np.array(trainrow[1:len(trainrow)-1])
            b = np.array(testrow[1:len(testrow)-1])
            # dist = distance.euclidean(a, b)
            dist1 = distance.cityblock(a,b)
            dist = np.linalg.norm(a-b)
            # print "distance for " , i , "= " ,dist
            result.append([dist , trainrow[len(trainrow)-1]])
            result1.append([dist1, trainrow[len(trainrow) - 1]])
            i+=1
        result.sort(key = lambda row: row[0:] )
        result1.sort(key=lambda row: row[0:])
        # print "after sorting = " , result
        for k in range(0, K) :
            minVal.append(result[k][1])
            minVal1.append(result[k][1])
        # print "top k elements" , minVal
        count = Counter(minVal);
        count1 = Counter(minVal1)
        # print " test ans =" , count.most_common()[0]
        # print 'type = ' , count.most_common()
        # print " test ans =", count.most_common()[0][0]
        # print "real ans = " , testrow[len(testrow)-1]
        if(count.most_common()[0][0] == testrow[len(testrow)-1]) :
            true+=1.0;
        else :
            false+=1.0
        if(count1.most_common()[0][0] == testrow[len(testrow)-1]) :
            correct+=1.0;
        else :
            wrong+=1.0;
        minVal = []
        minVal1 = []
    # print 'correct = ' , true
    # print 'wrong = ' , false
    print 'accuracy for K = ', K, "=" ,(true/(true+false))*100 , "%"


def trainingAccuracy(K,type) :
    minVal = [] ; minVal1 = [] ; true =0;false=0;correct=0;wrong=0;
    for i in range(0,len(trainMatrix)) :
        initial = np.array(trainMatrix[i])
        a = initial[1:len(initial) - 1]
        b = np.delete(trainMatrix,(i),axis =0)
        for j in range(0,len(b)) :
            c = np.array(b[j][1:len(b[j]) -1 ])
            Edist = distance.euclidean(a,c)
            Mdist = distance.cityblock(a,c)
            result3.append([Edist,b[j][len(b[j]) -1 ]])
            result4.append([Mdist,b[j][len(b[j]) -1 ]])
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
        # print " test ans =" , count.most_common()[0]
        # print 'type = ' , count.most_common()
        # print 'type1 = ', count1.most_common()
        # print " test ans =", count.most_common()[0][0]
        # print "real ans = " , testrow[len(testrow)-1]
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
    if(type ==  2) :
        print 'accuracy for K = ', K, "=", (true / (true + false)) * 100, "%"
    else :
        print 'accuracy for K = ', K, "=", (correct / (correct + wrong)) * 100, "%"

# print "Testing Accuracy --->"
# print "Euclidean Distance for KNN : "
# testingAccuracy(1);
# testingAccuracy(3);
# testingAccuracy(5);
# testingAccuracy(7);
# testingAccuracy(9);
# print "Manhattan Distance for KNN : "
# testingAccuracy(1);
# testingAccuracy(3);
# testingAccuracy(5);
# testingAccuracy(7);
# testingAccuracy(9);

print "-------------------"

print "Training Accuracy ----->"
print "Manhattan Distance for KNN : "
trainingAccuracy(1,1);
trainingAccuracy(3,1);
trainingAccuracy(5,1);
trainingAccuracy(7,1);
print "--------------------------"

print "Euclidean Distance for KNN : "

trainingAccuracy(1,2);
trainingAccuracy(3,2);
trainingAccuracy(5,2);
trainingAccuracy(7,2);
print "--------------------------"

