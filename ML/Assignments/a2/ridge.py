import numpy as np
import pandas as pd
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
# print trainDf


## 73 rows of testdata from X0 to Y ---------------------
testDf =  pd.DataFrame(testData)
testDf.columns = allHeaders
# print testDf

lambdaValue = [0.01,0.1,1]


def normalise(trainDf1):
    return ((pd.DataFrame(trainDf1[trainDf1.columns[1:15]])) - (pd.DataFrame(trainDf[trainDf.columns[1:15]])).mean())/((pd.DataFrame(trainDf[trainDf.columns[1:15]])).std())

train_norm = pd.DataFrame(normalise(trainDf))
train_norm.insert(0,'X0',1.0);
# print train_norm

trainMean = trainDf['Y'].mean()
trainStd = trainDf['Y'].std()
testMean = testDf['Y'].mean()
testStd = testDf['Y'].std()
# print trainMean
# print trainStd
# print testMean
# print testStd

test_norm = pd.DataFrame(normalise(testDf))
test_norm.insert(0,'X0',1.0)
# print test_norm

def parameters(newdf) :
    finalParams = []
    designMatrix = pd.DataFrame(newdf.drop('Y', 1))
    # print designMatrix
    designMatrix = np.array(designMatrix)
    # print designMatrix.shape
    designMatrixTranspose = designMatrix.transpose();
    # print designMatrixTranspose.shape
    Identity = np.identity(designMatrixTranspose.shape[0])
    Identity[0][0] = 0
    for i in lambdaValue:
        scalarId = i*Identity
        # print scalarId
        inverseValue = np.linalg.pinv((np.dot(designMatrixTranspose, designMatrix)) + scalarId )
        finalParams.append(np.dot(inverseValue, np.dot(designMatrixTranspose, newdf['Y'])))
    # print finalParams
    return finalParams


finalParams = parameters(train_norm)
# print finalParams[0]



def errorVal(X,Y,finalParams,mean,std) :
    actual = Y['Y']
    X.drop('Y',inplace=True)
    vectorX = np.array(X)
    pred = np.dot(finalParams.transpose(),vectorX)
    # pred = finalParams.transpose() * vectorX
    pred =  pred*std +mean
    # print abs(pred - actual)
    return abs(pred - actual)


trainerrors = []
trainMse = []
for j in finalParams :
    for i in range(0,len(train_norm)) :
        trainerrors.append(errorVal(train_norm.ix[i],trainDf.ix[i] , j , trainMean,trainStd))
    trainMse.append((np.sum(np.square(trainerrors))) / len(trainerrors))

print '--------------- RIDGE REGRESSION  --------------------'

print 'Training MSE : ' , (np.average(trainMse))
# finalParams=[]
# finalParams = parameters(test_norm)
# print finalParams[0]
testerrors = []
testMse = []

for j in finalParams :
    for i in range(0,len(test_norm)) :
        testerrors.append(errorVal(test_norm.ix[i],testDf.ix[i] , j,trainMean,trainStd))
    testMse.append((np.sum(np.square(testerrors))) / len(testerrors))

print 'Testing MSE : ', np.average(testMse)