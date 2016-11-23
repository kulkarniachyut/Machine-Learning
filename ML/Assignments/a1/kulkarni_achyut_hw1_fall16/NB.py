import pandas as pd
import numpy as np
import math as m
import time


trainMatrix = np.loadtxt(open("./train.txt","rb"),delimiter=",")
testMatrix = np.loadtxt(open("./test.txt","rb"),delimiter=",")
trainData = pd.read_csv("./train.csv" ,header=None)
# global var
# global avg

val = [1.0,1.0,1.0,-1000000.0,1.0,1.0,1.0]
logval = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

def priors() :
    type = []; count= [0,0,0,0,0,0,0] ;
    for i in range(0,len(trainMatrix)) :
        if((trainMatrix[i][len(trainMatrix[i])-1]) == 1 ):
            count[0]+=1.0;
        if((trainMatrix[i][len(trainMatrix[i])-1]) == 2) :
            count[1]+=1.0
        if ((trainMatrix[i][len(trainMatrix[i]) - 1]) == 3):
            count[2]+= 1.0
        if ((trainMatrix[i][len(trainMatrix[i]) - 1]) == 4):
            count[3]+= 1.0
        if ((trainMatrix[i][len(trainMatrix[i]) - 1]) == 5):
            count[4]+= 1.0
        if ((trainMatrix[i][len(trainMatrix[i]) - 1]) == 6):
            count[5]+= 1.0
        if ((trainMatrix[i][len(trainMatrix[i]) - 1]) == 7):
            count[6]+= 1.0
    for i in range(0,7) :
        type.append(count[i]/(len(trainMatrix)))
    return type
priorP = priors()

def calcN(j,test) :
    value = 1.0;val=1.0 ;
    ## test is a vector with 9-D =>> len =9
    for i in range(0,len(test)) :
        expo = 1.0;const = 1.0;
        if(var[i+1][j+1] != 0.0) :
            expo = m.exp(-((test[i] - avg[i+1][j+1])**2/(2*var[i+1][j+1])))
            const = ((1.0)/(m.sqrt(2*(m.pi)*(var[i+1][j+1]))))
            # print priorP[j]
            # print expo
            # print const
            logval[i] = m.log(expo) + m.log( const)
    logPrior = m.log(priorP[j])
    # print logval
    return (sum(logval) + logPrior)


def normalD(matrix) :
    count=0.0;
    # for i in range(0,len(testMatrix)) :
    for i in range(0, len(matrix)):
        # test = testMatrix[i][1:len(testMatrix[i])-1]
        test = matrix[i][1:len(matrix[i]) - 1]
        for j in range(0,7) :
            # print "j = " , j
            if(j != 3) :
                val[j] = calcN(j,test)
                # print val[j]
        # print val
        trainAns = np.argmax(val)
        if(trainAns !=3 ) :
            trainAns+=1

        # print "test data i = " ,i ,"class = ", trainAns , "real class = " ,matrix[i][len(matrix[i])-1]
        if(trainAns == matrix[i][len(matrix[i])-1]) :
            count+=1
    # print "Matrix Data Accuracy =  " , (((count)/(i))*100),"%"
    return (((count)/(i+1))*100)

mean= pd.DataFrame(trainData[trainData.columns[1:11]])
variance = pd.DataFrame(trainData[trainData.columns[1:11]])

avg = mean.groupby([10]).mean()
var = variance.groupby([10]).var()

# replace all 0's to ones in variance matrix
# var[var==0.0] = 1.0
# print avg
# print var
# print priorP

print "Calculating Testing Accuracy -----------------------------------------"
# start_time = time.time()
testAccuracy = normalD(testMatrix)
# print time.time() - start_time
print
print "Testing Data Accuracy = " , testAccuracy , "%"
print
print "Calculating Training Accuracy -----------------------------------------"
trainAccuracy = normalD(trainMatrix)
print
print "Training Data Accuracy = " , trainAccuracy, "%"
