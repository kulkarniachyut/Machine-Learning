import pandas as pd
import numpy as np


trainMatrix = np.loadtxt(open("/Users/achi/Desktop/train.txt","rb"),delimiter=",")
testMatrix = np.loadtxt(open("/Users/achi/Desktop/test.txt","rb"),delimiter=",")

# mean class matrix -- 9 features * 7 classes matrix to keep all means of all classes in one matrix
mean = np.zeros((9,7))
featureMatrix = np.zeros((len(trainMatrix),9))


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

def meancalc(feature) :
    type = [0,0,0,0,0,0,0]
    for i in range(0,len(trainMatrix)) :

        if(trainMatrix[i][len(trainMatrix[i])-1] == 1 ) :
            type[0]+=1
            mean[0][0]+= trainMatrix[i][1]
            mean[1][0]+= trainMatrix[i][2]
            mean[2][0]+= trainMatrix[i][3]
            mean[3][0]+= trainMatrix[i][4]
            mean[4][0]+= trainMatrix[i][5]
            mean[5][0]+= trainMatrix[i][6]
            mean[6][0]+= trainMatrix[i][7]
            mean[7][0]+= trainMatrix[i][8]
            mean[8][0]+= trainMatrix[i][9]
        if (trainMatrix[i][len(trainMatrix[i]) - 1] == 2):
            type[1]+=1
            mean[0][1]+= trainMatrix[i][1]
            mean[1][1]+= trainMatrix[i][2]
            mean[2][1]+= trainMatrix[i][3]
            mean[3][1]+= trainMatrix[i][4]
            mean[4][1]+= trainMatrix[i][5]
            mean[5][1]+= trainMatrix[i][6]
            mean[6][1]+= trainMatrix[i][7]
            mean[7][1]+= trainMatrix[i][8]
            mean[8][1]+= trainMatrix[i][9]
    print mean

def calculateMean() :
    # for i in range(0,len(trainMatrix)) :
    #     featureMatrix[i] = np.array(trainMatrix[i][1:len(trainMatrix[i])-1])

    for j in range(0,9) :
        for k in range(0,7) :
            for l in range(0,len(trainMatrix) ):
                print trainMatrix[l][len(trainMatrix[l])-1]
            print k
    return featureMatrix


# priorProb = priors()
# print priorProb

# meanMatrix = calculateMean()
# print meanMatrix

meancalc(1)
# meancalc(1)
# meancalc(1)
