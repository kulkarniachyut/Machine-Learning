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
lambdaValue = [0.0001,0.001,0.01,0.1,1,10]


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
        value = errorVal(test_norm.ix[i],cvtest.ix[i], finalParams, mean, std)
        # print  value
        testerrors.append(value)
        # testerrors.append(errorVal(test_norm.ix[i],cvtest.ix[i], finalParams, mean, std))
    mse = ((np.sum(np.square(testerrors))) / len(testerrors))
    # print 'mse for test : ' , ((np.sum(np.square(testerrors))) / len(testerrors))
    return mse

# kftotal = cross_validation.KFold(len(trainDf), n_folds=10, shuffle=True, random_state=1)

mseValues = [] ;mseValues2 = []; mseValues3  = [];mseValues6  = []
mseValues1 = [];mseValues4 = [];mseValues5  = []
totalTrain = np.array_split(trainDf,10)

for i in range(0,len(totalTrain)):
    cvTestDf = pd.DataFrame(totalTrain[i] , columns=allHeaders )
    cvTestDf =cvTestDf.reset_index(drop=True)
    # print cvTestDf
    # cvTestDf =
    # print cvTestDf
    cvTrainDf = pd.DataFrame(columns=allHeaders)
    for j in range(0,len(totalTrain)):
        if(j!=i) :
            # print totalTrain[j]
            cvTrainDf=cvTrainDf.append(totalTrain[j],ignore_index=True)
        # cvTrainDf = cvTrainDf.append(totalTrain[j])
    # print '--------------------------------------------------------------'
    # print cvTrainDf
    # print cvTestDf
    # print len(cvTrainDf)
    # print len(cvTestDf)
    print i
    # mseValues.append(ridge(cvTestDf, cvTrainDf, 0.0001))
    mseValue = ridge(cvTestDf, cvTrainDf, 0.0001)
    mseValues.append(mseValue)
    mseValue = ridge(cvTestDf, cvTrainDf, 0.001)
    mseValues1.append(mseValue)
    mseValue = ridge(cvTestDf, cvTrainDf, 0.01)
    mseValues2.append(mseValue)
    mseValue = ridge(cvTestDf, cvTrainDf, 0.1)
    mseValues3.append(mseValue)
    mseValue = ridge(cvTestDf, cvTrainDf, 1)
    mseValues4.append(mseValue)
    mseValue = ridge(cvTestDf, cvTrainDf, 10)
    mseValues5.append(mseValue)
    mseValue = ridge(cvTestDf, cvTrainDf, 1000)
    mseValues6.append(mseValue)
    # print 'lambda = ', 0.0001, ' Mean CV MSE : ', np.average(mseValues)
print 'lambda = ', 0.0001, ' Mean CV MSE : ',  np.average(mseValues)
print 'lambda = ', 0.001, ' Mean CV MSE : ',  np.average(mseValues1)
print 'lambda = ', 0.01, ' Mean CV MSE : ',  np.average(mseValues2)
print 'lambda = ', 0.1, ' Mean CV MSE : ',  np.average(mseValues3)
print 'lambda = ', 1, ' Mean CV MSE : ',  np.average(mseValues4)
print 'lambda = ', 10, ' Mean CV MSE : ',  np.average(mseValues5)
print 'lambda = ', 1000, ' Mean CV MSE : ',  np.average(mseValues6)
        # mseValues.append(ridge(cvTestDf, cvTrainDf, 0.0001))
