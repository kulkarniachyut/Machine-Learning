import numpy as np
import matplotlib.pyplot as plot
import scipy.stats
from scipy.stats import multivariate_normal
import pandas as pd
import random
from collections import Counter
import math
from scipy.spatial import distance
from scipy import cluster
from collections import defaultdict


blobDataFrame = pd.DataFrame();
blobDataFrame = pd.read_csv("hw5_blob(1).csv",header=None);

blobData = np.array(blobDataFrame)

def initialise(points,K):
    priors = [];stdev1=[];mean1=[]
    mean = random.sample(points,K)
    for i in range(K):
        priors.append(1.0/K)
    cov = np.identity(2)
    return {"mean" : mean, "priors" : priors , "stdev" : [1,1,1]}


initial = initialise(blobData,3)
# print initial


def getDensity(x, m,s):
    pdf = scipy.stats.multivariate_normal.pdf(x,mean=m,cov=s)
    return pdf


def getPosterior(x,priors,means,stdevs):
    posterior = {};sum_posteriors=0.0
    for k in range(0,len(priors)):
        posterior[k] = (getDensity(x, means[k],stdevs[k])) * priors[k]
    sum_posteriors = sum(posterior.values())
    for k in posterior.keys():
        posterior[k]/=sum_posteriors
    post = [posterior[0],posterior[1],posterior[2]]
    return post

def relearn(probabilities,points,originalMean):
    mean = [];std = [];a=[]
    a = np.sum(probabilities,axis=0)
    priors = a/len(probabilities)
    for i in range(0,3):
        prodVector=1.0;sumVector=0.0
        for j in range(0,len(points)):
            prodVector = probabilities[j][i] * points[j]
            sumVector+=prodVector
        sumVector = np.array(sumVector)/a[i]
        mean.append(sumVector)

    for i in range(0,3):
        prodVector=1.0;sumVector=0.0
        for j in range(0,len(points)):
            dotProd = np.dot((points[j] - originalMean[i]) ,(points[j] - originalMean[i]).transpose() )
            prodVector = probabilities[j][i] * dotProd
            sumVector+=prodVector
        sumVector = (sumVector)/a[i]
        # print sumVector[0]
        std.append(sumVector)
    # print "priors" , priors
    # print "mean", mean
    # print "std", std
    return (priors , mean,std)




def em(initial,points):
    # newPoints = []
    priors = initial["priors"]
    mean = np.array(initial["mean"])
    std = initial["stdev"]
    loop = True
    while(loop):
        print "priors" , priors
        prob = [] ; count=0
        for i in range(0,len(points)):
            prob.append((getPosterior(points[i],priors,mean,std)))
        (newPriors , newMean , newStd ) = relearn(np.array(prob) , points ,mean )

        for i in range(0,len(newPriors)):
            if(newPriors[i] == priors[i]):
                count+=1
            if(newStd[i] == std[i]):
                count+=1
        priors = newPriors
        mean = newMean
        std = newStd
        if(count == 6):
            loop = False


        # print "std", std

em(initial,blobData)
