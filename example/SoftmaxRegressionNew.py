from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from matplotlib import axis
np.random.seed(12)

def softmax(Z):
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

def softmax_stable(Z):
    e_Z = np.exp(Z - np.max(Z, axis= 1, keepdims= True))
    A = e_Z / e_Z.sum ( axis = 1, keepdims = True)
    return A

# cost and loss function
def softmax_loss(X, y, W):
    A = softmax_stable(X.dot(W))
    id0 = range(X.shape[0])

    return -np.mean(np.log(A[id0, y]))

def softmax_grad(X, y, W):
    A = softmax_stable(X.dot(W)) # shape of (N, C)
    id0 = range(X.shape[0])
    A[id0, y] -= 1 # A -Y, shape of (N, C)
    return X.T.dot(A) / X.shape[0]

def numerical_grad(X, Y, W, loss):
    eps = 1e-6
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i, j] += eps 
            W_n[i, j] -= eps
            g[i,j] = (loss(X, Y, W_p) - loss(X, Y, W_n))/(2*eps)
    return g

def softmax_fit(X, y, W, eta = 0.01, nepoches = 100, tol = 1e-5, batch_size = 10):
    W_old = W.copy()
    ep = 0
    lost_hist = [softmax_loss(X, y, W)]
    N = X.shape[0]
    nbatches = int(np.ceil(float(N)/batch_size)) # np.ceil(X) nhỏ nhất mà không nhỏ hơn X
    while ep < nepoches:
        ep += 1
        mix_ids = np.random.permutation(N)
        for i in range(nbatches):
            # lấy batch thứ i
            batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)]
            X_batch, y_batch = X[batch_ids], y[batch_ids]
            W = W - eta * softmax_grad(X_batch, y_batch, W)  # update GD
        lost_hist.append(softmax_loss(X, y, W))
        if np.linalg.norm(W - W_old)/ W.size < tol:
            break
        W_old = W.copy()

    return (W, lost_hist)
        


def pred(W, X):
    """
    predict output of each columns of X
    Class of each x_i is determined by location of max probability
    Note that class are indexed by [0, 1, 2, ...., C-1]
    """
    A = softmax_stable(X.dot(W))
    return np.argmax(A, axis = 1)


C = 5    # number of classes
N = 500  # number of points per class 
means = [[2, 2], [8, 3], [3, 6], [14, 2], [12, 8]]
cov = [[1, 0], [0, 1]]

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X3 = np.random.multivariate_normal(means[3], cov, N)
X4 = np.random.multivariate_normal(means[4], cov, N)

X = np.concatenate((X0, X1, X2, X3, X4), axis = 0) # each row is a datapoint
Xbar = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1) # bias trick 

y = np.asarray([0]*N + [1]*N + [2]*N+ [3]*N + [4]*N)

W_init = np.random.randn(Xbar.shape[1], C)
(W, loss_hist) = softmax_fit(Xbar, y, W_init, batch_size = 10, nepoches = 100, eta = 0.05)

def display(X, label):
    #     K = np.amax(label) + 1
    X0 = X[np.where(label == 0)[0]]
    X1 = X[np.where(label == 1)[0]]
    X2 = X[np.where(label == 2)[0]]
    X3 = X[np.where(label == 3)[0]]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'co', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'gs', markersize = 4, alpha = .8)
    plt.plot(X3[:, 0], X3[:, 1], 'y.', markersize = 4, alpha = .8)
    plt.plot(X4[:, 0], X4[:, 1], 'r*', markersize = 4, alpha = .8)
    plt.plot()
    plt.show()



xm = np.arange(-2, 18, 0.025)
xlen = len(xm)
ym = np.arange(-3, 11, 0.025)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)


# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# xx.ravel(), yy.ravel()

print(np.ones((1, xx.size)).shape)
xx1 = xx.ravel().reshape(-1, 1)
yy1 = yy.ravel().reshape(-1, 1)

# print(xx.shape, yy.shape)
XX = np.concatenate(( xx1, yy1, np.ones(( xx.size, 1))), axis = 1)


print(XX.shape)

Z = pred(W, XX)

Z = Z.reshape(xx.shape)
CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .1)


plt.xlim(-2, 18)
plt.ylim(-3, 11)
plt.xticks(())
plt.yticks(())
# plt.axis('equal')
display(X, y)
plt.show()