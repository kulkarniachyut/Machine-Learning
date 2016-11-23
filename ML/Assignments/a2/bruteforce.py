import numpy as np
import pandas as pd
import itertools
from sklearn.datasets import load_boston
boston = load_boston()
data = boston.data
trueOutput = boston.target

testData = []
trainData = []
dataFrame = pd.DataFrame(data)
dataFrame.columns = boston.feature_names
dataFrame.insert(0,'X0',1.0);
dataFrame['Y'] = pd.Series(trueOutput, index=dataFrame.index)
# print dataFrame
allHeaders = np.array(dataFrame.columns)

for i in range(0, 73):
    testData.append((np.array(dataFrame.ix[7*i])))
    dataFrame = dataFrame.drop([7 * i])


### 433 rows of train data from X0 to Y -----------------
trainData = np.array(dataFrame)
trainDf = pd.DataFrame(trainData)
trainDf.columns = allHeaders



## 73 rows of testdata from X0 to Y ---------------------
testDf =  pd.DataFrame(testData)
testDf.columns = allHeaders
# print testDf



select = boston.feature_names
comb = list(itertools.combinations(select, 4))
print len(comb)
print comb
# print trainDf
# for i in (comb):

newTrainDf = pd.DataFrame();
finalParamsArray = []
def getfinalParams(newTrainDf) :
    designMatrix = pd.DataFrame(newTrainDf.drop('Y', 1))
    # print designMatrix
    designMatrix = np.array(designMatrix)
    # print designMatrix.shape
    designMatrixTranspose = designMatrix.transpose();
    # print designMatrixTranspose.shape
    inverseValue = np.linalg.pinv(np.dot(designMatrixTranspose, designMatrix))

    finalParams = np.dot(inverseValue, np.dot(designMatrixTranspose, newTrainDf['Y']))
    finalParamsArray.append(finalParams)
    return finalParams


def errorVal(X,Y,mean,std,finalParams) :
    actual = Y['Y']
    X.drop('Y',inplace=True)
    vectorX = np.array(X)
    pred = np.dot(finalParams.transpose(),vectorX)
    # pred = finalParams.transpose() * vectorX
    pred =  pred*std + mean
    y = pred-actual
    # print mean , std
    # print abs(pred - actual)
    return abs(pred - actual)

def mse(newTrainDf):
    mean =  newTrainDf['Y'].mean()
    std =  newTrainDf['Y'].std()
    newTrainDf = (newTrainDf - newTrainDf.mean())/newTrainDf.std()
    newTrainDf.insert(0,'X0',1.0)
    finalParams = getfinalParams(newTrainDf)
    trainerrors = []
    testerrors = []
    for i in range(0, len(newTrainDf)):
        val = errorVal(newTrainDf.ix[i], trainDf.ix[i],mean,std,finalParams)
        trainerrors.append(errorVal(newTrainDf.ix[i], trainDf.ix[i], mean, std, finalParams))

    mse =  (np.sum(np.square(trainerrors)))/len(trainerrors)
    # print mse
    return mse

    # print newTrainDf
mseValues = []
mseComb = []
# msevalue = 10000000000.0
for j in comb :
    print '-------------------------'
    # print j
    newTrainDf[j[0]] = (trainDf[j[0]])
    newTrainDf[j[1]] = (trainDf[j[1]])
    newTrainDf[j[2]] = (trainDf[j[3]])
    newTrainDf[j[3]] = (trainDf[j[3]])
    newTrainDf['Y'] = trainDf['Y']
    msevalue = mse(newTrainDf)
    mseValues.append(msevalue)
    mseComb.append([j,msevalue])
    newTrainDf = pd.DataFrame();

mseValues = np.array(mseValues)
# print mseValues.argsort()[4:][::-1]
minIndex = np.argmin(mseValues)
# print mseValues
print mseComb
print mseComb[minIndex]
# print mseComb
# print newTrainDf






