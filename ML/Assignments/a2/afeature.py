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


pc = []
def pearsonCorrelation(X):
    npX = np.array(X)
    npY = np.array(trainDf.iloc[:,[14]])
    result = abs(np.float(np.dot(npX.transpose(),npY))/((np.sqrt(np.sum(np.square(npX)))) * (np.sqrt(np.sum(np.square(npY))))))
    pc.append(result)

for i in range(0,14):
    # print i
    X = trainDf.iloc[:,[i]]
    pearsonCorrelation(X)

# print len(pc)
pc = np.array(pc)
pc = pc.argsort()[-4:][::-1]
# print pc
# print trainDf
cols = ['CRIM','ZN', 'CHAS','NOX','AGE','DIS','RAD','TAX','B']

for i in range(0,len(cols)) :
    trainDf.drop(cols[i],axis=1,inplace=True)
    testDf.drop(cols[i], axis=1, inplace=True)

# print trainDf
# print testDf

def normalise(X) :
    return ((pd.DataFrame(X[X.columns[1:6]])) - (pd.DataFrame(trainDf[trainDf.columns[1:6]])).mean()) / ((pd.DataFrame(trainDf[trainDf.columns[1:6]])).std())

norm_train = pd.DataFrame(normalise(trainDf))
norm_train.insert(0,'X0',1.0);
# print norm_train

norm_test = pd.DataFrame(normalise(testDf))
norm_test.insert(0,'X0',1.0);
# print norm_test

designMatrix = pd.DataFrame(norm_train.drop('Y',1))
# print designMatrix
designMatrix = np.array(designMatrix)
# print designMatrix.shape
designMatrixTranspose = designMatrix.transpose();
# print designMatrixTranspose.shape


inverseValue = np.linalg.pinv(np.dot(designMatrixTranspose,designMatrix))

finalParams = np.dot(inverseValue,np.dot(designMatrixTranspose,norm_train['Y']))
# print finalParams


def errorVal(X,Y) :
    actual = Y['Y']
    X.drop('Y',inplace=True)
    vectorX = np.array(X)
    pred = np.dot(finalParams.transpose(),vectorX)
    # pred = finalParams.transpose() * vectorX
    pred =  pred*trainDf['Y'].std() +trainDf['Y'].mean()
    # print abs(pred - actual)
    return (pred - actual)


testerrors = []
for i in range(0,len(testDf)) :
    testerrors.append(errorVal(norm_test.ix[i],testDf.ix[i]))

print '-------------   FEATURE SELECTION WITH TOP 4 FEATURES      ------------'
trainerrors = []
for i in range(0,len(norm_train)) :
    trainerrors.append(errorVal(norm_train.ix[i],trainDf.ix[i]))

pc=[  'INDUS',  'RM'   ,'PTRATIO',     'LSTAT' ]

print 'The Top 4 features are : ' , \
    pc

print 'Training MSE : ',  (np.sum(np.square(trainerrors)))/len(trainerrors)

print 'Testing MSE : ',  (np.sum(np.square(testerrors)))/len(testerrors)
