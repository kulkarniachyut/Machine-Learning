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
    covMatrix = {}
    for i in range(0,3):
        covMatrix[i] = np.identity(2);
    return {"mean" : mean, "priors" : priors , "stdev" : covMatrix}


initial = initialise(blobData,3)


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
    return((post, sum_posteriors))

def relearn(probabilities,points,originalMean):
    mean = [];std = [];a=[];cov= {}
    a = np.sum(probabilities,axis=0)
    priors = a/len(probabilities)

    for i in range(0,3):
        prodVector=1.0;sumVector=np.array([0,0])
        for j in range(0,len(points)):
            prodVector = probabilities[j][i] * np.array(points[j])
            sumVector = sumVector + prodVector
        sumVector = np.array(sumVector)/a[i]
        mean.append(sumVector)

    for i in range(0,3):
        m = mean[i] ;sumvalue = np.zeros((2,2))
        for j in range(0,len(points)):
            sumvalue+= probabilities[j][i] * np.matrix((points[j] - m)).T * np.matrix((points[j] - m))
        sumvalue = sumvalue/a[i]
        cov[i] = sumvalue
    return (priors , mean,cov)

def em(initial,points):
    priors = initial["priors"]
    mean = np.array(initial["mean"])
    std = initial["stdev"]
    loop = True
    sumV = 0.0 ; oldsum = 0.0;count = 0;values = []
    while(loop):
        count+=1
        sumV
        prob = [] ;logLikelihood =0.0
        for i in range(0,len(points)):
            probty, sum_posteriors = getPosterior(points[i],priors,mean,std)
            prob.append(probty)
            logLikelihood+= np.log(sum_posteriors)
        (newPriors , newMean , newStd ) = relearn(np.array(prob) , points ,mean )
        print logLikelihood
        values.append(logLikelihood)
        if(round(logLikelihood , 6) == round(oldsum,6)):
            loop = False
        oldsum = logLikelihood
        priors = newPriors
        mean = newMean
        std = newStd
    print values    # break
    print count
    plot.plot(values)

def main():
    initial = initialise(blobData,3)
    em(initial,blobData)

for i in range(0,5):
    main();
plot.show()
