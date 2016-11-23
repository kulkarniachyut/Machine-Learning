import numpy as np
import matplotlib.pyplot as plot

predArr = [];
actualArr = []
lambdaValues = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def varFunc(predArr, actualArr):
    varVal = 0.0
    mean = (np.array(predArr)).mean()
    # print mean
    for i in range(0,len(predArr)):
        x = (predArr[i] - mean)**2
        varVal = varVal+ x;
    # print varVal
    return (varVal/len(predArr))

def biasFunc(predArr,actualArr):
    bias = 0.0
    mean = (np.array(predArr)).mean()
    # mean1 = (np.array(actualArr)).mean()
    # bias = (mean - mean1)**2

    for i in range(0, len(predArr)):
        x = (mean - actualArr[i]) ** 2
        bias = bias + x;
    # print bias
    return (bias / len(predArr))

def getSet(size):
    total = 100*size
    xpoints = np.random.uniform(-1,1,total)
    noise = np.random.normal(0,0.1,total)
    ypoints =[]
    for i in range(0,len(xpoints)):
        ypoints.append( 2*(xpoints[i]**2) + noise[i])
    dataset = { 'x' : xpoints , 'y' : ypoints}
    return  dataset

samples = 10
entireSet = getSet(samples)

def getParams(X,Y):
    inverse =np.linalg.pinv(np.dot(X.transpose(),X))
    finalParams = np.dot(inverse, np.dot(X.transpose(), Y ))
    # print finalParams
    return finalParams


def getParamsRidge(X,Y,L):
    Identity = np.identity(X.transpose().shape[0])
    Identity[0][0] = 0
    scalarId = L * Identity
    inverse =np.linalg.pinv(np.dot(X.transpose(),X) + scalarId)
    finalParams = ( np.dot(inverse, np.dot(X.transpose(), Y )))
    # print finalParams
    return finalParams

def errorVal(X,Y,finalParams,g):
    err = [];
    if(g ==1) :
        for i in range(0,len(X)):
            pred = 1.0
            predArr.append(pred)
            actualArr.append(Y[i])
            # print pred   , '--' , Y[i]
            err.append(abs(pred-Y[i]))
        # var = varFunc(predArr, actualArr)
        valueObj = {'pred' : predArr , 'actual' : actualArr , 'err' : ((np.sum(np.square(err)))/len(err))}
        # return ((np.sum(np.square(err)))/len(err))
        return valueObj
    if (g == 2):
        for i in range(0, len(X)):
            pred = finalParams[0]
            predArr.append(pred)
            actualArr.append(Y[i])
            # print pred, '--', Y[i]
            err.append(abs(pred - Y[i]))
        valueObj = {'pred': predArr, 'actual': actualArr, 'err': ((np.sum(np.square(err))) / len(err))}
        # print err
        # return ((np.sum(np.square(err))) / len(err))
        return valueObj
    if (g == 3):
        for i in range(0, len(X)):
            pred = np.dot(finalParams.transpose(),X[i])
            predArr.append(pred)
            actualArr.append(Y[i])
            # print pred, '--', Y[i]
            err.append(abs(pred - Y[i]))
        # print err
        valueObj = {'pred': predArr, 'actual': actualArr, 'err': ((np.sum(np.square(err))) / len(err))}
        # return ((np.sum(np.square(err))) / len(err))
        return valueObj
    if (g ==4 ):
        for i in range(0, len(X)):
            pred = np.dot(finalParams.transpose(), X[i])
            predArr.append(pred)
            actualArr.append(Y[i])
            # print pred, '--', Y[i]
            err.append(abs(pred - Y[i]))
        # print err
        valueObj = {'pred': predArr, 'actual': actualArr, 'err': ((np.sum(np.square(err))) / len(err))}
        # return ((np.sum(np.square(err))) / len(err))
        return valueObj
    if (g == 5):
        for i in range(0, len(X)):
            pred = np.dot(finalParams.transpose(), X[i])
            predArr.append(pred)
            actualArr.append(Y[i])
            # print pred, '--', Y[i]
            err.append(abs(pred - Y[i]))
        # print err
        valueObj = {'pred': predArr, 'actual': actualArr, 'err': ((np.sum(np.square(err))) / len(err))}
        return valueObj
        # return ((np.sum(np.square(err))) / len(err))
    if (g == 6):
        for i in range(0, len(X)):
            pred = np.dot(finalParams.transpose(), X[i])
            predArr.append(pred)
            actualArr.append(Y[i])
            # print pred, '--', Y[i]
            err.append(abs(pred - Y[i]))
        # print err
        valueObj = {'pred': predArr, 'actual': actualArr, 'err': ((np.sum(np.square(err))) / len(err))}
        # return ((np.sum(np.square(err))) / len(err))
        return valueObj


def RidgelinearReg(X,Y,L):
    X0 = np.ones(10)
    squareX = np.square(X)
    # NORMALISE  ???
    # X = (X - np.mean(X))/np.std(X)
    C = (np.vstack((X0,X,squareX))).transpose()
    finalParams = getParamsRidge(C,Y,L)
    # print finalParams
    # print C
    # err =[]
    err =(errorVal(C,Y,finalParams,4))
    return err


def linearReg(X,Y,g) :
    X0 = np.ones(10)
    if(g == 1 or g ==2 or g == 3) :
        # NORMALISE  ???
        # X = (X - np.mean(X))/np.std(X)
        C = (np.vstack((X0,X))).transpose()
        finalParams = getParams(C,Y)
        err = errorVal(C,Y,finalParams,g)
        return err
    squareX = np.square(X)
    cubeX = np.power(X,3)
    fourX = np.power(X,4)
    if(g == 4):
        C = (np.vstack((X0, X,squareX))).transpose()
        finalParams = getParams(C, Y)
        err = errorVal(C, Y, finalParams, g)
        return err
    if (g == 5):
        C = (np.vstack((X0, X, squareX,cubeX))).transpose()
        finalParams = getParams(C, Y)
        err = errorVal(C, Y, finalParams, g)
        return err
    if (g == 6):
        C = (np.vstack((X0, X, squareX,cubeX,fourX))).transpose()
        finalParams = getParams(C, Y)
        err = errorVal(C, Y, finalParams, g)
        return err



def main(g,reg, entireSet=entireSet):
    if(reg == 0):
        mse = []
        for j in range(0,100):
            X = [];Y=[];
            for i in range(0,10):
                X.append(entireSet['x'][j*10+i])
                Y.append(entireSet['y'][j*10 + i])
                # break
            result = linearReg(np.array(X),(np.array(Y)).transpose() , g)
            # break
            # print result
            mse.append(result['err'])
            # break
        # print 'g == ' , g , 'MSE : ' , (np.array(mse)).mean()
        # return result
        variance = varFunc(result['pred'] , result['actual'])
        bias = biasFunc(result['pred'], result['actual'])
        print 'Variance for G : ',g ,' :', variance
        print 'Bias for G :',g, " ;" ,bias
        predArr = [];
        actualArr = []
        # print mse
        plot.figure(g)
        plot.hist(mse,bins=10)
        plot.show()
        return (np.array(mse)).mean()
    else :

        for k in lambdaValues :
            mse = []
            for j in range(0, 100):
                X = [];
                Y = [];
                # mse = []
                for i in range(0, 10):
                    X.append(entireSet['x'][j * 10 + i])
                    Y.append(entireSet['y'][j * 10 + i])
                    # break

                result = RidgelinearReg(np.array(X), (np.array(Y)).transpose(),k)
                mse.append(result['err'])
            # return result
            variance = varFunc(result['pred'], result['actual'])
            bias = biasFunc(result['pred'], result['actual'])
            print 'Lambda Value', k ,'-----------------'
            print 'MSE : ' , (np.array(mse)).mean()
            print 'Variance for G:', g, ':', variance
            print 'Bias for G:', g, ":", bias
            predArr = [];
            actualArr = []
            # plot.figure(g)
            # plot.hist(mse, bins=10)
            # plot.show()
            # break
        return (np.array(mse)).mean()



def Linear():

    print'------------------------------------'
    print '------------------------------------'
    print 'For Data Set of 10 '
    print '------------------------------------'
    print '------------------------------------'
    mse  = main(1,0)
    print 'g == ' , 1 , 'MSE : ' , (np.array(mse))
    # variance = varFunc(result['predArr'], result['actualArray'])
    print '------------------------------------'
    mse  = main(2,0)
    print 'g == ' , 2 , 'MSE : ' , (np.array(mse))
    print '------------------------------------'
    mse  = main(3,0)
    print 'g == ' , 3 , 'MSE : ' , (np.array(mse))
    print '------------------------------------'
    mse  = main(4,0)
    print 'g == ' , 4 , 'MSE : ' , (np.array(mse))
    print '------------------------------------'
    mse  = main(5,0)
    print 'g == ' , 5 , 'MSE : ' , (np.array(mse))
    print '------------------------------------'
    mse  = main(6,0)
    print 'g == ' , 6 , 'MSE : ' , (np.array(mse))
    print '------------------------------------'
    print '------------------------------------'
    print 'For Data Set of 100 '

    samples = 100
    entireSet1 = getSet(samples)

    print '------------------------------------'
    print '------------------------------------'
    mse  = main(1,0,entireSet1)
    print 'g == ' , 1 , 'MSE : ' , (np.array(mse))
    print '------------------------------------'
    mse  = main(2,0,entireSet1)
    print 'g == ' , 2 , 'MSE : ' , (np.array(mse))
    print '------------------------------------'
    mse  = main(3,0,entireSet1)
    print 'g == ' , 3 , 'MSE : ' , (np.array(mse))
    print '------------------------------------'
    mse  = main(4,0,entireSet1)
    print 'g == ' , 4 , 'MSE : ' , (np.array(mse))
    print '------------------------------------'
    mse  = main(5,0,entireSet1)
    print 'g == ' , 5 , 'MSE : ' , (np.array(mse))
    print '------------------------------------'
    mse  = main(6,0,entireSet1)
    print 'g == ' , 6 , 'MSE : ' , (np.array(mse))
    print '------------------------------------'
    print '------------------------------------'

# mse  = main(1,0)
# print 'g == ' , 1 , 'MSE : ' , (np.array(mse))
Linear();
# samples = 100
# entireSet = getSet(samples)

def Regularised():

    mse = main(2, 1)

Regularised()
# print samples



