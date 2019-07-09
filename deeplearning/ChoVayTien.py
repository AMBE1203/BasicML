from __future__ import print_function
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# load data
data = pd.read_csv('./data/data_logistic.csv').values
N = data.shape[0]
d = data.shape[1]

print(d)
X = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)

#bias trick
Xbar = np.concatenate((np.ones((N, 1)), X), axis = 1)

# Vẽ data bằng scatter
plt.scatter(X[:10, 0], X[:10, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(X[10:, 0], X[10:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
plt.legend(loc=1)
plt.xlabel('mức lương (triệu)')
plt.ylabel('kinh nghiệm (năm)')
plt.show()