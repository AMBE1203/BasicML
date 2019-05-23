import numpy as np
from mnist import MNIST
import math
mdata = MNIST('./MNIST/')

mdata.load_testing()

def encodeOnehot(Y, C): 
    'chuyển Y sang dạng one-hot. Mỗi cột đại diện cho đầu ra của 1 điểm dữ liệu'
    Y0 = np.zeros((C, Y.shape[0]))
    for i in range(Y.shape[0]):
        Y0[Y[i], i] = 1
    return Y0


#khởi tạo dữ liệu
X_test = np.array(mdata.test_images).T/256.0
Y_test = np.array(mdata.test_labels)

#lấy 1000 điểm dữ liệu đầu tiên làm dữ liệu huấn luyện
X = X_test[:, :100]
Y = encodeOnehot(Y_test[:100], 10)
#softmax mới

def softmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

d0 = X.shape[0] 
d1 = 500
d2 = 10
theta = 0.001 

def MLP(X, Y, theta):
    
    W1 = np.random.randn(d0, d1)
    b1 = np.zeros((d1, 1))
    W2 = np.random.randn(d1, d2)
    b2 = np.zeros((d2, 1))
        
    #mini-batch GD
    nBatch = 50  #số lượng điểm dữ liệu mỗi batch
    nEpoch = 100 #số lượng epoch
        
    for e in range(nEpoch):
        id_rd = np.random.permutation(X.shape[1])
        for i in range(int(X.shape[1]/nBatch)):
            #lấy ra dữ liệu cho batch thứ i
            Xbatch = X[:, 0 : 0]
            Ybatch = Y[:, 0 : 0]
            for j in range(nBatch):
                id = id_rd[nBatch * i + j]
                Xbatch = np.concatenate((Xbatch, X[:, id : id + 1]), axis = 1)
                Ybatch = np.concatenate((Ybatch, Y[:, id : id + 1]), axis = 1)

                #feedforward
                Z1 = np.dot(W1.T, Xbatch) + b1
                A1 = np.maximum(Z1, 0)
                Z2 = np.dot(W2.T, A1) + b2
                Yhat = softmax(Z2)

                #backpropagation
                E2 = (Yhat - Ybatch)
                dW2 = np.dot(A1, E2.T)
                db2 = E2.sum(axis = 1, keepdims = True)

                E1 = np.dot(W2, E2)
                E1[Z1 <= 0] = 0

                dW1 = np.dot(Xbatch, E1.T)
                db1 = E1.sum(axis = 1, keepdims = True)

                #update 
                W2 = W2 - theta * dW2
                b2 = b2 - theta * db2
                W1 = W1 - theta * dW1
                b1 = b1 - theta * db1

    return (W1, b1, W2, b2)

(W1, b1, W2, b2) = MLP(X, Y, theta)

#dự đoán dữ liệu (dùng luôn dữ liệu huấn luyện)
Z1 = np.dot(W1.T, X) + b1
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
predicted_class = np.argmax(Z2, axis=0)
print('training accuracy: %.2f %%' % (100*np.mean(predicted_class == Y_test[:X.shape[1]])))