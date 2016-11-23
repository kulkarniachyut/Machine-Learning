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


def normalisetrain(trainDf):
    return ((pd.DataFrame(trainDf[trainDf.columns[1:15]])) - (pd.DataFrame(trainDf[trainDf.columns[1:15]])).mean())/((pd.DataFrame(trainDf[trainDf.columns[1:15]])).std())

df_norm = pd.DataFrame(normalisetrain(trainDf))
df_norm.insert(0,'X0',1.0);
# print df_norm

# print (pd.DataFrame(trainDf[trainDf.columns[1:15]])).mean()
# print (pd.DataFrame(trainDf[trainDf.columns[1:15]])).std()

def normalisetest(test):
    return ((pd.DataFrame(test[test.columns[1:15]])) - (pd.DataFrame(trainDf[trainDf.columns[1:15]])).mean())/((pd.DataFrame(trainDf[trainDf.columns[1:15]])).std())

test_norm = pd.DataFrame(normalisetest(testDf))
test_norm.insert(0,'X0',1.0)
# print test_norm

designMatrix = pd.DataFrame(df_norm.drop('Y',1))
# print designMatrix
designMatrix = np.array(designMatrix)
# print designMatrix.shape
designMatrixTranspose = designMatrix.transpose();
# print designMatrixTranspose.shape


inverseValue = np.linalg.pinv(np.dot(designMatrixTranspose,designMatrix))

finalParams = np.dot(inverseValue,np.dot(designMatrixTranspose,df_norm['Y']))
# print finalParams
# print finalParams.shape

def errorVal(X,Y) :
    actual = Y['Y']
    X.drop('Y',inplace=True)
    vectorX = np.array(X)
    pred = np.dot(finalParams.transpose(),vectorX)
    # pred = finalParams.transpose() * vectorX
    pred =  pred*trainDf['Y'].std() +trainDf['Y'].mean()
    # print abs(pred - actual)
    return (pred - actual)

def errorVal1(X,Y) :
    actual = Y['Y']
    # print actual
    X.drop('Y',inplace=True)
    vectorX = np.array(X)
    # print vectorX
    pred = np.dot(finalParams.transpose(),vectorX)
    pred =  pred*trainDf['Y'].std() +trainDf['Y'].mean()
    # print abs(pred - actual)
    return abs(pred - actual)


testerrors = []
for i in range(0,len(testDf)) :
    testerrors.append(errorVal1(test_norm.ix[i],testDf.ix[i]))

print '--------------- LINEAR REGRESSION  --------------------'
trainerrors = []
for i in range(0,len(df_norm)) :
    trainerrors.append(errorVal(df_norm.ix[i],trainDf.ix[i]))



print 'Training MSE : ',  (np.sum(np.square(trainerrors)))/len(trainerrors)

print 'Testing MSE : ',  (np.sum(np.square(testerrors)))/len(testerrors)

