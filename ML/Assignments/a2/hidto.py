import matplotlib.pyplot as plot
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

trainDf = trainDf.drop('X0',1)
trainDf = trainDf.drop('Y',1)

for i in boston.feature_names :
    plot.figure(i)
    plot.hist(trainDf[i],bins=10)
plot.show()
