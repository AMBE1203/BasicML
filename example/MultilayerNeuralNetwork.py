from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12)


d0 = 2 # data dimension
C = 3 # number of classes
N = 100 # number of points per class

X = np.zeros((N*C, d0)) #data matrix ( each row == single example)
y = np.zeros(N*C, dtype='uint8') # class labels

for j in range(C):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
# lets visualize the data:
# plt.scatter(X[:N, 0], X[:N, 1], c=y[:N], s=40, cmap=plt.cm.Spectral)

plt.plot(X[:N, 0], X[:N, 1], 'bs', markersize = 7, markeredgecolor = 'k');
plt.plot(X[N:2*N, 0], X[N:2*N, 1], 'ro', markersize = 7, markeredgecolor = 'k');
plt.plot(X[2*N:, 0], X[2*N:, 1], 'g^', markersize = 7, markeredgecolor = 'k');
# plt.axis('off')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
plt.show()

def softmax_stable(Z):

    '''
    Compute softmax values for each sets of scores in Z.
    each row of Z is a set of scores.    
    '''
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

def crossentropy_loss(Yhat, y):
    '''
    Yhat: a numpy array of shape (Npoints, nClasses) -- predicted output
    y: a numpy array of shape (Npoints) -- ground truth.
    NOTE: We don’t need to use the one-hot vector here since most of elements
    are zeros. When programming in numpy, in each row of Yhat, we need to access
    to the corresponding index only.
    '''
    id0 = range(Yhat.shape[0])
    return -np.mean(np.log(Yhat[id0, y]))


def mlp_init(d0, d1, d2):
    '''
    Initialize W1, b1, W2, b2
    d0: dimension of input data
    d1: number of hidden unit
    d2: number of output unit = number of classes
    '''
    W1 = 0.01*np.random.randn(d0, d1)
    b1 = np.zeros(d1)
    W2 = 0.01*np.random.randn(d1, d2)
    b2 = np.zeros(d2)
    return (W1, b1, W2, b2)

def mlp_predict(X, W1, b1, W2, b2):
    '''
    Suppose that the network has been trained, predict class of new points.
    X: data matrix, each ROW is one data point.
    W1, b1, W2, b2: learned weight matrices and biases
    '''
    Z1 = X.dot(W1) + b1 # shape (N, d1)
    A1 = np.maximum(Z1, 0) # shape (N, d1)
    Z2 = A1.dot(W2) + b2 # shape(N, d2)
    return np.argmax(Z2, axis = 1)

def mlp_fit(X, y, W1, b1, W2, b2, eta, N):
    loss_hist = []
    for i in range(20000): # number of epoches
        # feedforward
        Z1 = X.dot(W1) + b1 # shape (N, d1)
        A1 = np.maximum(Z1, 0) # shape (N, d1)
        Z2 = A1.dot(W2) + b2 # shape(N, d2)
        Yhat = softmax_stable(Z2)

        if i % 1000 ==0: # print loss after each 1000 iterations
            loss = crossentropy_loss(Yhat, y)
            print("iter %d, loss: %f" %(i, loss))
            loss_hist.append(loss)

        # back propagation
        id0 = range(Yhat.shape[0])
        Yhat[id0, y] -= 1
        E2 = Yhat /N # shape (N, d2)
        dW2 = np.dot(A1.T, E2) # shape (d1, d2)
        db2 = np.sum(E2, axis = 0) #shape (d2, )
        E1 = np.dot(E2, W2.T) # shape (N, d1)
        E1[Z1 <= 0 ] = 0 # gradient of ReLu, shape (N, d1)
        dW1 = np.dot(X.T, E1) # shape (d0, d1)
        db1 = np.sum(E1, axis = 0) # shape (d1, )

        # gradient descent update
        W1 += -eta*dW1
        b1 += - eta*db1
        W2 += -eta*dW2
        b2 += -eta*db2

    return (W1, b1, W2, b2, loss_hist)



d0 = 2
d1 = h = 500 # size of hidden layer
d2 = C = 3
eta = 1


(W1, b1, W2, b2) = mlp_init(d0, d1, d2)
(W1, b1, W2, b2, loss_hist) = mlp_fit(X, y, W1, b1, W2, b2, eta, N)
y_pred = mlp_predict(X, W1, b1, W2, b2)
acc = 100*np.mean(y_pred == y)
print('training accuracy: %.2f %%' %acc)
    
# visualize results
xm = np.arange(-1.5, 1.5, 0.025)
xlen = len(xm)
ym = np.arange(-1.5, 1.5, 0.025)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)

xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

X0 = np.vstack((xx1, yy1)).T
Z = mlp_predict(X0, W1, b1, W2, b2)
Z = Z.reshape(xx.shape)

plt.clf()
CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .3)



plt.plot(X[:N, 0], X[:N, 1], 'bs', markersize = 7, markeredgecolor = 'k');
plt.plot(X[N:2*N, 0], X[N:2*N, 1], 'go', markersize = 7, markeredgecolor = 'k');
plt.plot(X[2*N:, 0], X[2*N:, 1], 'r^', markersize = 7, markeredgecolor = 'k');

plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xticks(())
plt.yticks(())
str0 = '#hidden units = ' + str(d1) + ', accuracy =' + str(acc) + '%'
plt.title(str0, fontsize = 15)
plt.show()