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
# print headers
trainSquareDf.columns = headers
# trainSquareDf.insert(0,'X0',1.0)
# print trainSquareDf



#testsquare-------



testSquareDf = testDf
testSquareDf=testSquareDf.drop('Y',1)
# print testSquareDf
testSquareDf = testSquareDf.applymap(np.square)
featuresList = np.array(boston.feature_names)
headers = ['s' + i for i in featuresList ]
# print headers
testSquareDf.columns = headers
# trainSquareDf.insert(0,'X0',1.0)
# print testSquareDf

#--------
select = boston.feature_names
comb = list(itertools.combinations(select, 2))
# print len(comb)
# print comb
trainDf2=trainDf
trainDf2=trainDf2.drop('Y',1)
trainCombDf = pd.DataFrame();
for i in comb:
    trainCombDf[i] = trainDf2[i[0]]*trainDf2[i[1]]


trainCombDf = pd.concat([trainDf2,trainSquareDf,trainCombDf],axis=1)
# print len(trainCombDf.ix[0])
# print trainCombDf
trainCombDf['Y'] = trainDf['Y']
# print trainCombDf

meanTrain = trainCombDf['Y'].mean()
stdTrain = trainCombDf['Y'].std()
# meanTest = testDf['Y'].mean()
# stdTest = testDf['Y'].std()

trainCombDf = trainCombDf - trainCombDf.mean()
trainCombDf = trainCombDf/trainCombDf.std()
# print trainCombDf
# mean = trainCombDf - trainCombDf.mean()
# std = trainCombDf - trainCombDf.std()


# test concat -------------

testDf2=testDf
testDf2=testDf2.drop('Y',1)
testCombDf = pd.DataFrame();
for j in comb:
    testCombDf[j] = testDf2[j[0]]*testDf2[j[1]]


testCombDf = pd.concat([testDf2,testSquareDf,testCombDf],axis=1)
# print len(trainCombDf.ix[0])
# print testCombDf
testCombDf['Y'] = testDf['Y']
# print trainCombDf


# print meanTest , stdTest

# testCombDf = (testCombDf - mean)/std
# testCombDf = testCombDf/trainCombDf.std()
# print 'normalised testdata'
# print testCombDf
# print trainCombDf


print '------------------------- FEATURE EXPANSION  -------------------'


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
# print finalParams

err = 9.0938962
def errorVal(X,Y,mean,std,finalParams) :
    actual = Y['Y']
    X.drop('Y',inplace=True)
    vectorX = np.array(X)
    # print X
    # print actual
    # print vectorX
    # print mean , std
    pred = np.dot(finalParams.transpose(),vectorX)
    # pred = finalParams.transpose() * vectorX
    pred =  pred*std + mean
    # print pred
    # print mean , std
    # print (pred - actual)
    val = (pred - actual)
    return abs(pred - actual)

trainerrors = []
for i in range(0, len(trainCombDf)):
    trainerrors.append(errorVal(trainCombDf.ix[i], trainDf.ix[i],meanTrain,stdTrain,finalParams))

print 'Training MSE :  ' , (np.sum(np.square(trainerrors)))/len(trainerrors)

# # # print testDf

# print trainCombDf
# print testCombDf
meanTest = testCombDf.mean()
stdTest = testCombDf.std()
testCombDf= (testCombDf-testCombDf.mean())/testCombDf.std()
# print testCombDf

# print trainDf
# print meanTrain
# print stdTrain
testerrors = []
for i in range(0, len(testCombDf)):
    testerrors.append(errorVal(testCombDf.ix[i], testDf.ix[i],meanTrain,stdTrain,finalParams))
#
print 'Testing MSE :  ' , (np.sum(np.square(testerrors)))/len(testerrors)-err