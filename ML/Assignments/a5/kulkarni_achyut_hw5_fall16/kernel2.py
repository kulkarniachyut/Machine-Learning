import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import random
import math
import scipy
from scipy.spatial import distance
from scipy import cluster
from collections import defaultdict


blobDataFrame = pd.DataFrame();
blobDataFrame = pd.read_csv("hw5_blob(1).csv",header=None);
circleDataFrame = pd.DataFrame();
circleDataFrame = pd.read_csv("hw5_circle(1).csv",header=None);

blobData = np.array(blobDataFrame)
circleData = np.array(circleDataFrame)



def centroid(indices,points):
    sum_x =0;sum_y =0;
    for i in indices:
        sum_x=sum_x+ (points[i][0])
        sum_y =sum_y + np.sum(points[i][1])
    return [sum_x/len(indices) , sum_y/len(indices)]

cluster =[[],[]]

def kFuncDist(points,i , clusterIndices , j , kMatrix ):
    kval = kMatrix[i][i]
    totalPoints = len(clusterIndices[j])
    distance = (kMatrix[i,clusterIndices[j]]).sum()
    distance = distance/totalPoints
    val = (kMatrix[clusterIndices[j],:][:,clusterIndices[j]]).sum()
    val = val/(totalPoints**2)
    dist = ((kval - (2*distance) + val))
    return dist



def kmeans(K,points,cluster,kMatrix):
    dict= defaultdict(list)
    loop = True
    count = 0;
    while(loop):
        count+=1
        dist=[[],[]];newcluster = [[],[]]
        for j in range(0,len(cluster)):
            for  i in range(0,len(points)):
                dist[j].append(kFuncDist(points, i ,  cluster, j ,kMatrix))
        for z in range(0,len(points)):
            if(float(dist[0][z]) > float(dist[1][z])):
                newcluster[1].append(z)
            else:
                newcluster[0].append(z)
        # import pdb; pdb.set_trace()
        if(np.array_equal(newcluster[0],cluster[0])):
            loop = False
        cluster[0] = newcluster[0]
        cluster[1] = newcluster[1]
        for i in range(0,len(newcluster)):
            x=[];y=[]
            for j in newcluster[i]:
                x.append(points[j][0])
                y.append(points[j][1])
            plot.scatter(x,y,color=np.random.rand(3,1))
    plot.show()


def getCluster(points):
    for i in range(0,250):
        cluster[1].append(i)
        cluster[0].append(499-i)
    return cluster


def kernelFunction(a,b):
    val = ((a*b)).sum(axis=1) + (2*(b**4 ).sum() * 2*(a**4).sum(axis=1))
    return val

def getKMatrix(points):
    matrix = np.zeros((500,500))
    for i in range(0,len(points)):
        matrix[i,:] = kernelFunction(points  , points[i])
    return matrix


def __main__():
    print " ---------Circle Data ------"
    kMatrix = getKMatrix(circleData)
    initialCluster = getCluster(circleData);
    kmeans(2, circleData , initialCluster,kMatrix)


__main__()
