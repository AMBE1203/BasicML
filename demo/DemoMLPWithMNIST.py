from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt 
from mnist import MNIST

mdata = MNIST('./MNIST/')
mdata.load_testing()

# khởi tạo dữ liệu
X_test = np.array(mdata.test_images)/256.0
Y_test = np.array(mdata.test_labels)


# lấy 1000 điểm dữ liệu đầu tiên làm dữ liệu huấn luyện. mỗi điểm dữ liệu là 1 hàng
X = X_test[:1000, :] # shape (1000, 784)
y = Y_test[:1000]  # shape (1000, )



def softmax_stable(Z):
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
    A = e_Z/ e_Z.sum(axis = 1, keepdims = True)
    return A

def crossentropy_loss(Yhat, y):
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
    Z1 = X.dot(W1) + b1
    A1 = np.maximum(Z1, 0) 
    Z2 = A1.dot(W2) + b2 
    return np.argmax(Z2, axis = 1)

def mlp_fit(X, y, W1, b1, W2, b2, eta):
    loss_hist = []
    batch_size = 50 # số lượng điểm dữ liệu mỗi batch
    nEpoch = 100 # số lượng epoch
    ep = 0
    N = X.shape[0]
    nBatch = int(np.ceil(float(N)/batch_size)) # np.ceil(X) nhỏ nhất mà không nhỏ hơn X

    while ep < nEpoch:
        ep += 1
        mix_ids = np.random.permutation(N)
       
        for i in range(nBatch):
            # lấy dữ liệu cho batch thứ i
            batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)]
            Xbatch, ybatch = X[batch_ids], y[batch_ids]


            # feed forward
            Z1 = Xbatch.dot(W1) + b1
            A1 = np.maximum(Z1, 0)
            Z2 = A1.dot(W2) + b2
            Yhat = softmax_stable(Z2)

            # backpropagation
            id0 = range(Yhat.shape[0])
            Yhat[id0, ybatch] -= 1
            E2 = Yhat / batch_size # bỏ chia trung bình thì accuracy = 100%, thêm vào thì giảm xuống
            dW2 = np.dot(A1.T, E2)
            db2 = np.sum(E2, axis = 0)

            E1 = np.dot(E2, W2.T)
            E1[Z1 <= 0] = 0
            dW1 = np.dot(Xbatch.T, E1) # shape (d0, d1)
            db1 = np.sum(E1, axis = 0) # shape (d1, )

            # gradient descent update
            W1 += -eta*dW1
            b1 += - eta*db1
            W2 += -eta*dW2
            b2 += -eta*db2  
        

    return (W1, b1, W2, b2, loss_hist)


d0 = X.shape[1] # số chiều của dữ liệu
d1 = 500
d2 = 10 # số class

eta = 0.001

(W1, b1, W2, b2) = mlp_init(d0, d1, d2)
(W1, b1, W2, b2, loss_hist) = mlp_fit(X, y, W1, b1, W2, b2, eta)
y_pred = mlp_predict(X, W1, b1, W2, b2)
print(y_pred)
print(y)
acc = 100*np.mean(y_pred == y)
print('training accuracy: %.2f %%' %acc)





    
    
        



