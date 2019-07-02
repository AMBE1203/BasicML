from __future__ import print_function
import numpy as np
from cs231n.data_utils import load_CIFAR10

cifar10_dir = './data/cifar-10-batches-py/'



X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 1000)

# mean image of all trainning image
img_mean = np.mean(X_train, axis = 0)

def feature_enginer(X):
    X -= img_mean # zero-centered
    N = X.shape[0] # number of data point
    X = X.reshape(N, -1)
    return np.concatenate((X, np.ones((N, 1))), axis = 1) # bias trick

X_train = feature_enginer(X_train)
X_val = feature_enginer(X_val)
X_test = feature_enginer(X_test)

print('X_train shape: ', X_train.shape)
print('X_val shape: ', X_val.shape)
print('X_test shape: ', X_test.shape)


