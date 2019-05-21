from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12)

'''
Mỗi cột là 1 điểm dữ liệu
'''
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# extended data 
X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)
print(X)

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logistic_sigmoid_regression(X, y, w_init, eta, nepoches = 10000): #eta == learning rate
    w = [w_init]
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    ep = 0
    check_w_after = 20
    while ep < nepoches:
        #mix data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)  # X[:, i] lấy về cột thứ i (nhưng kết quả trả về là 1 ma trận hàng có shape (1, d) nên phải reshape về cột) 
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            ep += 1
            # stopping criteria
            if ep % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < 1e-4:
                    return w
            w.append(w_new)

    return w

eta = .05
d = X.shape[0]
w_init = np.random.randn(d, 1)

w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w[-1])
# w0 = w[-1][0]
# w1 = w[-1][1]
# với kết quả tìm được, đầu ra y có thể dự đoán được theo công thức y=sigmoid(w0 + w1*x)
# Với dữ liệu trong tập trainning kết quả là
print(sigmoid(np.dot(w[-1].T, X)))

# display output
X0 = X[1, np.where( y == 0)][0]
y0 = y[np.where(y == 0)]
X1 = X[1, np.where( y == 1)][0]
y1 = y[np.where(y == 1)]

plt.plot(X0, y0, 'ro', markersize = 8)
plt.plot(X1, y1, 'bs', markersize = 8)

xx = np.linspace(0, 6, 1000) # lấy 1000 điểm nằm trong khoảng [0, 6]
w0 = w[-1][0][0]
w1 = w[-1][1][0]

threshold = -w0/w1 # ngưỡng quyết định đỗ trượt
yy = sigmoid(w0 + w1*xx)
plt.axis([-2, 8, -1, 2])
plt.plot(xx, yy, 'g-', linewidth = 2)
plt.plot(threshold, .5, 'y^', markersize = 8)
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.show()

'''
Logistic regression với weight decay (là kỹ thuật phổ biến tránh overfitting, nó là 1 kỹ thuật regularization, trong đó 1 đại lượng
tỷ lệ với bình phương norm2 của vector hệ số được cộng vào hàm mất mát để hạn chế độ lớn của các hệ số)
'''
'''
Mỗi hàng là 1 điểm dữ liệu
'''

# hàm ước lượng xác suất cho mỗi điểm dữ liệu
def prob(w, X):
    return sigmoid(X.dot(w))

# hàm tính giá trị hàm mất mát với weight decay
def loss(w, X, y, lam):
    z = prob(w, X)
    return -np.mean(y*np.log(z) + (1-y)*np.log(1-z)) + 0.5*lam/X.shape[0] * np.sum(w*w)

def logistic_regression_with_decay(w_init, X, y, lam = 0.001, eta = 0.1, nepoches = 2000):
    N, d = X.shape[0], X.shape[1]
    w =  w_old = w_init
    loss_hist = [loss(w_init, X, y, lam)] # store history of loss in loss_hist
    ep = 0
    while ep < nepoches:
        ep += 1
        mix_ids = np.random.permutation(N)
        for i in mix_ids:
            xi = X[i]
            yi = y[i]
            zi = sigmoid(xi.dot(w))
            w = w - eta*((zi-yi)*xi + lam*w)
            loss_hist.append(loss(w, X, y, lam))
            if np.linalg.norm(w - w_old)/d < 1e-6:
                break
            w_old = w

    return (w, loss_hist)

X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
N = X.shape[0]

# bias trick
Xbar = np.concatenate((X, np.ones((N, 1))), axis = 1)
print(Xbar.shape)

w_init = np.random.randn(Xbar.shape[1])
lam = 0.0001
w, loss_hist = logistic_regression_with_decay(w_init, Xbar, y, lam, eta = 0.05, nepoches = 500)
print('Solution of Logistic Regression: ',w)
print('Final loss: ',loss(w, Xbar, y, lam))
