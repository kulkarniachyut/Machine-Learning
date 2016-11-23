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
# dataFrame.insert(0,'X0',1.0);
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
# print trainDf


## 73 rows of testdata from X0 to Y ---------------------
testDf =  pd.DataFrame(testData)
testDf.columns = allHeaders
# print testDf

trainSquareDf = trainDf
trainSquareDf=trainSquareDf.drop('Y',1)
# print trainSquareDf
trainSquareDf = trainSquareDf.applymap(np.square)
featuresList = np.array(boston.feature_names)
headers = ['s' + i for i in featuresList ]
print headers
trainSquareDf.columns = headers
# trainSquareDf.insert(0,'X0',1.0)
# print trainSquareDf


select = boston.feature_names
comb = list(itertools.combinations(select, 2))
print len(comb)
# print comb
trainDf2=trainDf
trainDf2=trainDf2.drop('Y',1)
trainCombDf = pd.DataFrame();
for i in comb:
    trainCombDf[i] = trainDf2[i[0]]*trainDf2[i[1]]


trainCombDf = pd.concat([trainDf2,trainSquareDf,trainCombDf],axis=1)
print len(trainCombDf.ix[0])
# print trainCombDf
trainCombDf['Y'] = trainDf['Y']
# print trainCombDf

mean = trainCombDf['Y'].mean()
std = trainCombDf['Y'].std()

trainCombDf = trainCombDf - trainCombDf.mean()
trainCombDf = trainCombDf/trainCombDf.std()
print trainCombDf

def getfinalParams(newTrainDf):
    designMatrix = pd.DataFrame(newTrainDf.drop('Y', 1))
    # print designMatrix
    designMatrix = np.array(designMatrix)
    # print designMatrix.shape
    designMatrixTranspose = designMatrix.transpose();
    # print designMatrixTranspose.shape
    inverseValue = np.linalg.pinv(np.dot(designMatrixTranspose, designMatrix))
    finalParams = np.dot(inverseValue, np.dot(designMatrixTranspose, newTrainDf['Y']))
    return finalParams

finalParams= getfinalParams(trainCombDf)
print finalParams


def errorVal(X,Y,mean,std,finalParams) :
    actual = Y['Y']
    X.drop('Y',inplace=True)
    vectorX = np.array(X)
    pred = np.dot(finalParams.transpose(),vectorX)
    # pred = finalParams.transpose() * vectorX
    pred =  pred*std + mean
    # print mean , std
    # print abs(pred - actual)
    return abs(pred - actual)

trainerrors = []
for i in range(0, len(trainCombDf)):
    trainerrors.append(errorVal(trainCombDf.ix[i], trainDf.ix[i],mean,std,finalParams))

print 'mse :  ' , (np.sum(np.square(trainerrors)))/len(trainerrors)