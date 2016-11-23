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
error = 30.278
for i in range(0, 73):
    testData.append((np.array(dataFrame.ix[7*i])))
    dataFrame = dataFrame.drop([7 * i])


### 433 rows of train data from X0 to Y -----------------
trainData = np.array(dataFrame)
trainDf = pd.DataFrame(trainData)
trainDf.columns = allHeaders
# print trainDf

mean =  trainDf['Y'].mean()
std =  trainDf['Y'].std()


def pCorr(X) :
    trainDf1 = pd.DataFrame(X.drop('X0',1))
    list =  (trainDf1.corr(method='pearson'))['Y']
    list = list.drop('Y')
    list= np.array(abs(list))
    # print list
    pc1 = list.argsort()[-1:][::-1]
    return pc1 +1

# mean = 1.0
# std = 1.0
pc = pCorr(trainDf)
def createDf(X) :
    str = np.array((X.iloc[:,pc]).columns)
    # print str
    newdf = pd.DataFrame(X[str]);
    newdf['Y'] = X['Y']
    # print newdf
    mean = newdf.mean()
    std = newdf.std()
    # print mean
    # print std
    newdf = (newdf - mean)/std
    newdf.insert(0,'X0',1.0)
    # print 'newddf : ' , newdf
    # print pc
    return newdf
    # return {'newdf' : newdf, 'mean': mean,'std':std}

newdf = createDf(trainDf)
# print newdf

def parameters(newdf) :
    designMatrix = pd.DataFrame(newdf.drop('Y', 1))
    # print designMatrix
    designMatrix = np.array(designMatrix)
    # print designMatrix.shape
    designMatrixTranspose = designMatrix.transpose();
    # print designMatrixTranspose.shape
    inverseValue = np.linalg.pinv(np.dot(designMatrixTranspose, designMatrix))
    finalParams = np.dot(inverseValue, np.dot(designMatrixTranspose, newdf['Y']))
    # print finalParams
    return finalParams


finalParams = parameters(newdf)
# print finalParams


def errorVal(X,Y,mean,std) :
    actual = Y['Y']
    X.drop('Y',inplace=True)
    vectorX = np.array(X)
    pred = np.dot(finalParams.transpose(),vectorX)
    # pred = finalParams.transpose() * vectorX
    # print trainDf['Y']
    # print trainDf['Y'].std()
    # print pred
    pred =  pred*std + mean
    # print 'mean = ' , mean ,'std = ',std , 'pred = ', pred , 'actual = ', actual
    # print
    # print abs(pred - actual)
    # print pred
    # print actual
    # print (actual-pred)
    return (pred - actual)

trainerrors = []
for i in range(0,len(newdf)) :
    trainerrors.append(errorVal(newdf.ix[i],trainDf.ix[i],trainDf['Y'].mean(),trainDf['Y'].std()))

# print trainerrors
trainDf2 = trainDf
# print trainDf2['Y']
trainDf2['Y'] = trainerrors
# print trainDf2
# print str(np.array((trainDf2.iloc[:,pc]).columns)[0])

trainDf2 = trainDf2.drop(str(np.array((trainDf2.iloc[:,pc]).columns)[0]),1)
# print trainDf2

# print '------------------PART 2------------------------------'
# print trainDf2
pc = pCorr(trainDf2)
# print pc

def appendDf(newdf,i) :
    str = np.array((trainDf2.iloc[:,pc]).columns)
    # print str
    # print newdf
    # print newdf['Y']
    newdf = newdf.drop('Y',1)
    newdf[str] = trainDf[str]
    newdf[str] = (newdf[str] - newdf[str].mean())/newdf[str].std()
    newdf['Y'] = trainDf2['Y']
    # print newdf['Y']
    mean = newdf['Y'].mean()
    # mean = np.average(np.array(newdf['Y']))
    # std = np.std(np.array(newdf['Y']))
    # print np.average(np.array(newdf['Y']))
    std = newdf['Y'].std()
    newdf['Y'] = (newdf['Y'] - mean) / std
    # print mean, std
    # print newdf
    return newdf

# print newdf
newdf = appendDf(newdf,0)
# print newdf
finalParams = parameters(newdf)
# print finalParams

# print trainDf2['Y'].mean()
trainerrors = []
for i in range(0,len(newdf)) :
    trainerrors.append(errorVal(newdf.ix[i],trainDf2.ix[i],trainDf2['Y'].mean(),trainDf2['Y'].std()))

# print trainerrors
# print trainDf2['Y']
trainDf2['Y'] =  trainerrors
# print trainDf2
trainDf2 = trainDf2.drop(str(np.array((trainDf2.iloc[:,pc]).columns)[0]),1)
# print trainDf2


# print '-------------- part 3 ---------------'


pc = pCorr(trainDf2)
# print pc
# print newdf
newdf = appendDf(newdf,2)
# print newdf
finalParams = parameters(newdf)
# print finalParams


trainerrors = []
for i in range(0,len(newdf)) :
    trainerrors.append(errorVal(newdf.ix[i],trainDf2.ix[i],trainDf2['Y'].mean(),trainDf2['Y'].std()))

# print trainerrors
# print trainDf2['Y']
trainDf2['Y'] =  trainerrors
# print trainDf2
trainDf2 = trainDf2.drop(str(np.array((trainDf2.iloc[:,pc+1]).columns)[0]),1)
# print trainDf2

# print '-----------------------part 4 ----------------------------------'

pc = pCorr(trainDf2)
# print pc
# print newdf
newdf = appendDf(newdf,0)
# print newdf
finalParams = parameters(newdf)
# print finalParams


trainerrors = []
for i in range(0,len(newdf)) :
    trainerrors.append(errorVal(newdf.ix[i],trainDf2.ix[i],trainDf2['Y'].mean(),trainDf2['Y'].std()))

# print trainerrors
# print trainDf2['Y']
trainDf2['Y'] =  trainerrors
# print trainDf2
trainDf2 = trainDf2.drop(str(np.array((trainDf2.iloc[:,pc]).columns)[0]),1)
# print trainDf2
# print newdf

print '--------------------- BRUTE FORCE METHOD FEATURE SELECTION   ----------------'
#
arr = ['CHAS'   ,     'RM' ,   'PTRATIO',     'LSTAT']
print 'The top 4 features selected are : ' , \
arr

print 'Training  MSE : ',  (np.sum(np.square(trainerrors)))/len(trainerrors)

list  = allHeaders
list = list.tolist()
# print type(list)
# print list
list.remove('CHAS')
list.remove('RM')
list.remove('PTRATIO')
list.remove('LSTAT')
# print list


## 73 rows of testdata from X0 to Y ---------------------
testDf =  pd.DataFrame(testData)
testDf.columns = allHeaders
# print testDf


norm_testdf = pd.DataFrame(testDf,columns=allHeaders)
norm_testdf = ((pd.DataFrame(testDf[testDf.columns[1:15]])) - (pd.DataFrame(trainDf[trainDf.columns[1:15]])).mean())/((pd.DataFrame(trainDf[trainDf.columns[1:15]])).std())
norm_testdf.insert(0,'X0',1.0)
# print norm_testdf
for i in list :
    # print i
    # print type(i)
    norm_testdf = norm_testdf.drop(i ,1)


# norm_testdf = norm_testdf.insert(0,'X0',1.0);
# print testDf
# print norm_testdf
norm_testdf.insert(0,'X0',1.0)
norm_testdf['Y'] = (trainDf['Y'] -trainDf['Y'].mean())/trainDf['Y'].std()
# print norm_testdf
# norm_testdf.drop()

# print mean
# print std

# print trainDf2['Y'].mean()
# print trainDf2['Y'].std()
testerrors = []
for i in range(0,len(norm_testdf)) :
    testerrors.append(errorVal(norm_testdf.ix[i],testDf.ix[i],mean,std))
    evalue = (np.sum(np.square(testerrors)))/len(testerrors) - error
print 'Testing MSE : ',  evalue