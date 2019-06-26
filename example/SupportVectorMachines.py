from __future__ import print_function
import numpy as np 
# simulated sample
means = [[2, 1], [4, 2]]
cov = [[.3, .2], [.2, .3]]

N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # blue class data
X1 = np.random.multivariate_normal(means[1], cov, N) # red class data
X = np.concatenate((X0, X1), axis = 0) # all data
y = np.concatenate((np.ones(N), -np.ones(N)), axis = 0) # label

# solving the dual problem (variable: lamda)
from cvxopt import matrix, solvers

V = np.concatenate((X0, -X1), axis = 0) # V in the book
Q = matrix(V.dot(V.T))
p = matrix(-np.ones((2*N, 1)))   # objective function 1/2 lambda^T*Q*lambda - 1^T*lambda

# build A,b, G, h
G = matrix(-np.eye(2*N))
h = matrix(np.zeros((2*N, 1)))
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros((1, 1)))
solvers.options['show_progress'] = False
sol = solvers.qp(Q, p, G, h, A, b)
l = np.array(sol['x']) # solution lambda
# calculate w and b
w = V.T.dot(l)
S = np.where(l > 1e-8)[0] # support set, 1e-8 to avoid small value of l.
b = np.mean(y[S].reshape(-1, 1) - X[S,:].dot(w))
print('Number of suport vectors = ', S.size)
print('w = ', w.T)
print('b = ', b)


# solution by sklearn
from sklearn.svm import SVC

model = SVC(kernel = 'linear', C = 1e5)
model.fit(X, y)

w = model.coef_
b = model.intercept_

print('Solution by sklearn: ')


print('w = ', w)
print('b = ', b)
