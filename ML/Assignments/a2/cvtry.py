import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn import cross_validation
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
lambdaValue = [0.0001,0.001,0.01,0.1,1,10,100]



totalTrain = np.array(trainDf)

# print totalTrain
# print len(totalTrain)


def normalise(test,train):
    # return (test - train.mean())/train.std()
    return ((pd.DataFrame(test[test.columns[1:15]])) - (pd.DataFrame(train[train.columns[1:15]])).mean())/((pd.DataFrame(train[train.columns[1:15]])).std())


def getParameters(cvtrain,v):
    # finalParams = []
    designMatrix = pd.DataFrame(cvtrain.drop('Y', 1))
    # print designMatrix
    designMatrix = np.array(designMatrix)
    # print designMatrix.shape
    designMatrixTranspose = designMatrix.transpose();
    # print designMatrixTranspose.shape
    Identity = np.identity(designMatrixTranspose.shape[0])
    Identity[0][0] = 0
    scalarId = v * Identity
    # print scalarId
    inverseValue = np.linalg.pinv((np.dot(designMatrixTranspose, designMatrix)) + scalarId)
    # print (np.dot(inverseValue, np.dot(designMatrixTranspose, cvtrain['Y'])))
    finalParams = (np.dot(inverseValue, np.dot(designMatrixTranspose, cvtrain['Y'])))
    # finalParams.append(np.dot(inverseValue, np.dot(designMatrixTranspose, cvtrain['Y'])))
    return finalParams


def errorVal(X,Y,finalParams,mean,std) :
    actual = Y['Y']
    X.drop('Y',inplace=True)
    vectorX = np.array(X)
    pred = np.dot(finalParams.transpose(),vectorX)
    # pred = finalParams.transpose() * vectorX
    pred =  pred*std +mean
    # print abs(pred - actual)
    return abs(pred - actual)


def ridge(cvtest,cvtrain,v) :
    # print cvtrain
    # print cvtest
    # print cvtrain['Y'].mean() ,cvtrain['Y'].std()
    mean = cvtrain['Y'].mean()
    std = cvtrain['Y'].std()
    test_norm = normalise(cvtest,cvtrain)
    train_norm = normalise(cvtrain,cvtrain)
    test_norm.insert(0, 'X0', 1.0);
    train_norm.insert(0, 'X0', 1.0);
    # print test_norm
    # print train_norm
    finalParams = getParameters(train_norm,v);
    testerrors = []
    for i in range(0, len(test_norm)):
        testerrors.append(errorVal(test_norm.ix[i],cvtest.ix[i], finalParams, mean, std))
    mse = ((np.sum(np.square(testerrors))) / len(testerrors))
    # print 'mse for test : ' , ((np.sum(np.square(testerrors))) / len(testerrors))
    return mse

kftotal = cross_validation.KFold(len(trainDf), n_folds=10, shuffle=True, random_state=5)


mseValues = []
for train, test in kftotal:
    # print train, '\n', test, '\n\n'
    for v in lambdaValue:
        cvTestDf = pd.DataFrame(columns=allHeaders)
        cvTrainDf = pd.DataFrame(columns=allHeaders)
        for i in test:
            series = pd.Series(trainDf.ix[i])
            cvTestDf = cvTestDf.append(series, ignore_index=True)
        for i in train:
            series = pd.Series(trainDf.ix[i])
            cvTrainDf = cvTrainDf.append(series, ignore_index=True)
        # print cvTestDf
        # print len(cvTestDf)
        # print len(cvTrainDf)
        # print v
        mseValues.append(ridge(cvTestDf,cvTrainDf,v))
        # break
    mseValues = np.array(mseValues)
    print 'lambda = ' , v , ' Mean CV MSE : ' ,np.average(mseValues)
    # break