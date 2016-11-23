import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import random
from scipy.spatial import distance
from scipy import cluster
from collections import defaultdict


blobDataFrame = pd.DataFrame();
blobDataFrame = pd.read_csv("hw5_blob(1).csv",header=None);
circleDataFrame = pd.DataFrame();
circleDataFrame = pd.read_csv("hw5_circle(1).csv",header=None);
# print circleDataFrame


blobData = np.array(blobDataFrame)
squareBlob = np.square(blobData)
circleData = np.array(circleDataFrame)
# squareCircle = np.square(circleData)
squareCircle = np.power((circleData) , 2)
squareDf = pd.DataFrame(data=squareCircle);
# print squareDf
XY = 1.414 * squareDf[0]*squareDf[1];
squareDf[2]=XY
squareDf[3]=squareDf[1]
squareDf[1]= squareDf[2]
del squareDf[3]
print squareDf
# print XY
# print squareDf
newCircleData = np.array(squareDf)


# clusters = [[1,2,3],[0.5,0.6,-0.3],[0.3,1,2]]
def centroid(indices,points):
    sum_x =0;sum_y =0;sum_z=0;
    for i in indices:
        sum_x=sum_x+ np.sum(points[i][0])
        sum_y =sum_y + np.sum(points[i][1])
        sum_z = sum_z + np.sum(points[i][2])
    return [sum_x/len(indices) , sum_y/len(indices),sum_z/len(indices)]


def kmeans(K,points,originalPoints):
    clusters = random.sample(points,K)

    print clusters
    # clusters = [[1,1],[5,7]]
    dict= defaultdict(list)
    # print clusters
    dist=[];
    loop = True
    while(loop):
        count=0;old=[]
        for i in range(0,len(points)):
            for j in range(0,len(clusters)):
                dist.append(distance.euclidean(points[i], clusters[j]));
            k = np.argmin(dist)
            dict[k].append(i)
            dist = []

        for i in range(0,len(clusters)):
            old.append(clusters[i])

        for i in range(0,len(clusters)):
            print clusters[i]
            print "dict" , dict[i]
            clusters[i] = centroid(dict[i],points)
            print clusters[i]
        for i in range(0, len(clusters)):
            if(distance.euclidean(old[i],clusters[i]) == 0):
                count+=1
        # print count
        x = []
        for i in range(0,K):
            print dict[i]
            print len(dict[i])
            x.append(dict[i])

        dict = defaultdict(list)
        if(count == len(clusters)):
            loop = False
            for i in range(0, len(x)):
                xPoints = [];
                yPoints = [];
                for j in x[i]:
                    xPoints.append(originalPoints[j][0])
                    yPoints.append(originalPoints[j][1])
                print i
                plot.scatter(xPoints, yPoints, color=np.random.rand(3, 1))
            plot.show()


def __main__():
    print " ---------Circle Data ------"
    print "For K= ", 2 , "--------------------"
    # kmeans(2,squareCircle,circleData)
    kmeans(2, newCircleData, circleData)
    print



__main__()
