from __future__ import print_function
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

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

def svm_loss_naive(W, X, y, reg):
    '''
    Calculate loss and gradient of the loss function at W. Naive way
    W: 2d numpy array of shape (d, C). The weight matrix.
    X: 2d numpy array of shape (N, d). The training data
    y: 1d numpy array of shape (N,). The training label
    reg: a positive number. The regularization parameter
    '''
    d, C, N =100, W.shape[1], X.shape[0] # data dim, number of classes, number of point
    loss = 0
    dW = np.zeros_like(W)
    for n in range(N):
        xn = X[n]
        score = xn.dot(W)
        for j in range(C):
            if j == y[n]:
                continue
            margin = 1 - score[y[n]] + score[j]
            if margin > 0:
                loss +=  margin
                dW[:, j] += xn
                dW[: , y[n]] -= xn
        
    loss /= N
    loss += 0.5*reg*np.sum(W*W)
    dW /= N
    dW += reg*W
    return loss, dW

# random, small data
d, C, N = 100, 3, 300
reg = .1
W_rand = np.random.rand(d, C)
X_rand = np.random.rand(N, d)
y_rand = np.random.randint(0, C, N)

# sanity check
print('Loss with reg = 0 : ', svm_loss_naive(W_rand, X_rand, y_rand, 0)[0])
print('Loss with reg = 0.1 : ', svm_loss_naive(W_rand, X_rand, y_rand, 0.1)[0])


# more efficient way to compute loss and grad
def svm_loss_vectorized(W, X, y, reg):
    d, C = 3073, W.shape[1]
    N = X.shape[0]
    loss = 0
    dW = np.zeros_like(W)

    Z = X.dot(W) # shape of (N, C)
    id0 = np.arange(Z.shape[0])
    correct_class_score = Z[id0, y].reshape(N, 1) # shape of (N, 1)
    margins = np.maximum(0, Z- correct_class_score + 1) # shape of (N, C)
    margins[id0, y] = 0
    loss = np.sum(margins)
    loss /= N
    loss += 0.5 * reg* np.sum(W*W)

    F = (margins > 0).astype(int) # shape of (N, C)
    F[np.arange(F.shape[0]), y] = np.sum(-F, axis = 1)
    dW = X.T.dot(F)/N + reg * W
    return loss, dW

d, C = 3073, 10
W_rand = np.random.randn(d, C)
import time
t1 = time.time()
l1, dW1 = svm_loss_naive(W_rand, X_train, y_train, reg)
t2 = time.time()
l2, dW2 = svm_loss_vectorized(W_rand, X_train, y_train, reg)
t3 = time.time()
print('Naive -- run time:', t2 - t1, '(s)')
print('Vectorized -- run time:', t3 - t2, '(s)')
print('loss difference:', np.linalg.norm(l1 - l2))
print('gradient difference:', np.linalg.norm(dW1 - dW2))



# mini-batch gradient descent

def multiclass_svm_GD(X, y, Winit, reg, lr =.1, batch_size = 1000, num_iters = 50, print_every = 10):
    W = Winit
    loss_history = []
    for it in range(num_iters):
        mix_ids = np.random.permutation(X.shape[0])
        n_batches = int(np.ceil(X.shape[0]/float(batch_size)))
        for ib in range(n_batches):
            ids = mix_ids[batch_size*ib:min(batch_size*(ib+1), X.shape[0])]
            X_batch, y_batch = X[ids], y[ids]

            lossib, dW = svm_loss_vectorized(W, X_batch, y_batch, reg)
            loss_history.append(lossib)
            W -= lr*dW
        if it % print_every == 0 and it >0:
            print('it %d/%d, loss = %f' %(it, num_iters, loss_history[it]))

    return W, loss_history

d, C = X_train.shape[1], 10
reg = .1
Winit = 0.00001*np.random.randn(d, C)
W, loss_history = multiclass_svm_GD(X_train, y_train, Winit, reg, lr = 1e-8, batch_size = 1000, num_iters = 50, print_every = 5)

def multi_svm_predict(W, X):
    Z = X.dot(W)
    return np.argmax(Z, axis = 1)

def evaluate(W, X, y):
    y_pred = multi_svm_predict(W, X)
    acc = 100*np.mean(y_pred == y)
    return acc


'''
Việc tiếp theo là sử dụng tập validation để chọn ra các bộ tham số mô hình phù hợp. Có hai
tham số trong thuật toán tối ưu multi-class SVM: regularization và learning rate. Hai tham
số này sẽ được tìm dựa trên các cặp giá trị cho trước. Bộ giá trị khiến cho độ chính xác của
mô hình trên tập validation cao nhất sẽ được dùng để đánh giá tập kiểm thử
'''
lrs = [1e-9, 1e-8, 1e-7, 1e-6]
regs = [0.1, 0.01, 0.001, 0.0001]
best_W = 0
best_acc = 0
for lr in lrs:
    for reg in regs:
        W, loss_history = multiclass_svm_GD(X_train, y_train, Winit, reg, lr=1e-8, batch_size=1000, num_iters=100, print_every=1e20)
        acc = evaluate(W, X_val, y_val)
        print('lr = %e, reg = %e, loss = %f, validation acc = %.2f' %(lr, reg, loss_history[-1], acc))
        if acc > best_acc:
            best_acc = acc
            best_W = W

acc = evaluate(best_W, X_test, y_test)
print('Accuracy on test data = %2f %%' %acc)

# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_history)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()