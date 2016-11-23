import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
data = boston.data
trueOutput = boston.target
dataFrame = pd.DataFrame(data)
pc = []


def pearsonCoefficient(X) :
    result = ((np.dot(X, trueOutput)) * 1.0) / ((np.sqrt(np.sum(np.square(X)))) * (np.sqrt(np.sum(np.square(trueOutput)))))
    pc.append(result)
    return



def main() :
    for i in range(0,13) :
        X = dataFrame[i]
        pearsonCoefficient(X)

main();

print len(pc)

