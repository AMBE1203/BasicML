from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse 
from matplotlib import axis
np.random.seed(12)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# each col is a datapoint
X = np.concatenate((X0, X1, X2), axis = 0).T

# extended data
X = np.concatenate((np.ones((1, 3*N)), X), axis = 0)

C = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T



def display(X, label):
    X0 = X[:, label == 0]
    X1 = X[:, label == 1]
    X2 = X[:, label == 2]

    plt.plot(X0[0, :], X0[1, :], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[0, :], X1[1, :], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[0, :], X2[1, :], 'rs', markersize = 4, alpha = .8)

    plt.axis('off')
    plt.plot()
    plt.show()

display(X[1:, :], original_label) # chỉ lấy từ hàng 1 trở đi, bỏ bias trick



'''
one-hot coding chỉ đúng 1 phần tử của vector nhãn yi = 1, các phần tử còn lại bằng 0
ví dụ với y = [0, 2, 1, 0] và 3 class thì ta sẽ có 
            [[1, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0]]
'''

def conver_labels(y, C = C):
    Y = sparse.coo_matrix((np.ones_like(y),
    (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

def softmax(Z):
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A

def softmax_stable(Z):
    '''
    Compute softmax values for each sets of scores in Z.
    each column of Z is a set of score.    
    Tránh hiện tượng tràn số khi exp(z) quá lớn
    '''
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    A = e_Z / e_Z.sum(axis = 0)
    return A

def softmax_regression(X, y, W_init, eta, tol = 1e-4, nepoches = 10000 ):
    W = [W_init]
    C = W_init.shape[1]
    Y = conver_labels(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]

    ep = 0
    check_w_after = 20
    while ep < nepoches:
        # mix data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax(np.dot(W[-1].T, xi))
            W_new = W[-1] + eta*xi.dot((yi - ai).T)
            ep += 1
            # stopping criteria
            if ep%check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:  # check_w_after này có tác dụng là sau khoảng 1 số lần epoch thì sẽ kiểm tra xem là w gần như đã min chưa ý.
                    return W
            W.append(W_new)
    return W # W[-1] is the solution, W is all history weights

    

# hàm dự đoán class cho dữ liệu mới
def pred(W, X):
    A = softmax_stable(W.T.dot(X))
    return np.argmax(A, axis = 0)

eta = .05
d = X.shape[0]
W_init = np.random.randn(d, C)
W = softmax_regression(X, original_label, W_init, eta)
print(W[-1])

xm = np.arange(-2, 11, 0.025)
xlen = len(xm)
ym = np.arange( -3, 10, 0.025)
ylen = len(ym)

xx, yy = np.meshgrid(xm, ym) # tạo 1 lưới hình chữ nhật trên các giá trị của 2 mảng xm, ym
xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size) # Ravel trả về một mảng một chiều. Bản sao được thực hiện chỉ khi cần thiết.

XX = np.concatenate((np.ones((1, xx.size)), xx1, yy1), axis = 0)
print(XX.shape)
Z = pred(W[-1], XX)

print(Z)

# Put the result into a color plot
Z = Z.reshape(xx.shape)

CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .1)

plt.xlim(-2, 11)
plt.ylim(-3, 10)
plt.xticks(())
plt.yticks(())
display(X[1:, :], original_label)
plt.show()

'''
SoftmaxRegression với thư viện Sklearn
'''
from mnist import MNIST
from sklearn import linear_model
from sklearn.metrics import accuracy_score
 
mntrain=MNIST('./MNIST/')
mntrain.load_training()
Xtrain=np.asanyarray(mntrain.train_images)/255.0
ytrain=np.array(mntrain.train_labels.tolist())


mntest=MNIST('./MNIST/')
mntest.load_testing()
Xtest=np.asarray(mntest.test_images)/255.0
ytest=np.array(mntest.test_labels.tolist())

# train:
logreg=linear_model.LogisticRegression(C=1e5, solver='lbfgs',multi_class='multinomial') # sử dụng sofrmax regression
logreg.fit(Xtrain,ytrain)
#test:
y_pred=logreg.predict(Xtest)
print ("Accuracy: %.2f %%" %(100*accuracy_score(ytest, y_pred.tolist())))

