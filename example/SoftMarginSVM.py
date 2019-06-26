from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

means = [[2, 2], [4, 2]]
cov = [[.7, 0], [0, .7]]

N = 20 # number of sample class
X0 = np.random.multivariate_normal(means[0], cov, N) # each row is a data point
X1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X0, X1))
y = np.concatenate((np.ones(N), -np.ones(N)))

# solution by sklearn
from sklearn.svm import SVC
C = 100
clf = SVC(kernel = 'linear', C = C)
clf.fit(X, y)
W_skleran = clf.coef_.reshape(-1, 1)
b_sklearn = clf.intercept_[0]

print(W_skleran.T, b_sklearn)

# solution by CVXOPT
from cvxopt import matrix, solvers
# build K
V = np.concatenate((X0, -X1), axis = 0) # V[n, :] = y[n] * X[n]
K = matrix(V.dot(V.T))
p = matrix(-np.ones((2*N, 1)))

# build A, b, G, h
G = matrix(np.vstack((-np.eye(2*N), np.eye(2*N))))
h = np.vstack((np.zeros((2*N, 1)), C*np.ones((2*N, 1))))
h = matrix(np.vstack((np.zeros((2*N, 1)), C*np.ones((2*N, 1)))))
A = matrix(y.reshape((-1, 2*N)))
b = matrix(np.zeros((1, 1))) # continue on next page
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)
l = np.array(sol['x']).reshape(2*N) # lambda vector
# support set
S = np.where(l > 1e-5)[0]
S2 = np.where(l < .999*C)[0]
# margin set
M = [val for val in S if val in S2] # intersection of two lists
VS = V[S] # shape (NS, d)
lS = l[S] # shape (NS,)
w_dual = lS.dot(VS) # shape (d,)
yM = y[M] # shape(NM,)
XM = X[M] # shape(NM, d)
b_dual = np.mean(yM - XM.dot(w_dual)) # shape (1,)
print('w_dual = ', w_dual)
print('b_dual = ', b_dual)

# solution with gradient descent

lam = 1./C
def loss(X, y, w, b):
    '''
    X.shape = (2N, d), y.shape= (2N, ), w.shape = (d, ), b is a scalar
    '''
    z = X.dot(w) + b # shape (2N, )
    yz = y*z
    return (np.sum(np.maximum(0, 1 - yz)) + .5*lam*w.dot(w))/X.shape[0]

def grad(X, y, w, b):
    z = X.dot(w) + b
    yz = y*z
    active_set = np.where(yz <= 1)[0] # consider 1 -yz >= 0 only
    _yX = -X*y[:, np.newaxis] # each row is y_n*x_n
    grad_w = (np.sum(_yX[active_set], axis = 0) + lam*w)/X.shape[0]
    grad_b = (-np.sum(y[active_set]))/X.shape[0]

    return (grad_w, grad_b)

def num_grad(X, y, w, b):
    eps = 1e-10
    gw = np.zeros_like(w)
    gb = 0

    for i in range(len(w)):
        wp = w.copy()
        wm = w.copy()
        wp[i] += eps
        wm[i] -= eps
        gw[i] = (loss(X, y, wp, b) - loss(X, y, wm, b))/(2*eps)
    
    gb = (loss(X, y, w, b+eps) - loss(X, y, w, b - eps))/(2*eps)
    return (gw, gb)

w = .1*np.random.randn(X.shape[1])
b = np.random.randn()
(gw0, gb0) = grad(X, y, w, b)
(gw1, gb1) = num_grad(X, y, w, b)
print('grad_w difference = ', np.linalg.norm(gw0 - gw1))
print('grad_b difference = ', np.linalg.norm(gb0 - gb1))


def softmarginSVM_grad(X, y, w0, b0, eta):
    w = w0
    b = b0
    it = 0
    while it < 10000:
        it += 1
        (gw, gb) = grad(X, y, w, b)
        w -= eta*gw
        b -= eta*gb
        if (it % 1000) == 0:
            print('iter %d' %it + ' loss: %f' %loss(X, y, w, b))

    return (w, b)

w0 = .1*np.random.randn(X.shape[1])
b0 = .1*np.random.randn()
lr = 0.05
(w_hinge, b_hinge) = softmarginSVM_grad(X, y, w0, b0, lr)
print('w_hinge = ', w_hinge)
print('b_hinge = ', b_hinge)
