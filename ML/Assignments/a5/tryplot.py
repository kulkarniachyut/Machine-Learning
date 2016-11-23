import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


blobDataFrame = pd.DataFrame();
blobDataFrame = pd.read_csv("hw5_blob(1).csv",header=None);
circleDataFrame = pd.DataFrame();
circleDataFrame = pd.read_csv("hw5_circle(1).csv",header=None);

blobData = np.array(blobDataFrame)
circleData = np.array(circleDataFrame)
# print circleData

N = 50
x = np.array(blobDataFrame[0])
y = np.array(blobDataFrame[1])
print x
print y
# print y
# x = np.random.rand(N)
# y = np.random.rand(N)

plt.scatter(x, y)
plt.show()
