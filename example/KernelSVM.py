from __future__ import print_function
import numpy as np 
from sklearn import svm

# XOR data and targets
X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y = np.array([0, 0, 1, 1])

# fit the model
for kernel in ('sigmoid', 'pol', 'rbf'):
    clf = svm.SVC(kernel = kernel, gamma = 4, coef0 = 0)
    clf.fit(X, y)

    