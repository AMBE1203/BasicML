from __future__ import print_function
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

'''
Gradient Descent với hàm 1 biến
'''

def grad(x):
    return 2*x + 5*np.cos(x)

def cost(x):
    return x ** 2 + 5*np.sin(x)

def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = myGD1(-5, .1)
print('Solution x1 = %f, cost = %f, after %d iteration' %(x1[-1], cost(x1[-1]),it1))

# tạo 1000 điểm dữ liệu gần đường thẳng y = 4 + 3x rồi dùng thư viện sckit-learn để tìm nghiệm
X = np.random.rand(1000, 1)
y = 4 + 3*X + .5*np.random.randn(1000) # noise added
model = LinearRegression()
model.fit(X.reshape(-1,1), y)
w, b = model.coef_[0][0], model.intercept_[0]
sol_sklearn = np.array([b,w])
print('Solution found by sklearn: ',sol_sklearn)


'''
Gradient Descent với hàm nhiều biến
'''
X = np.random.rand(1000, 1)
y = 4 + 3*X + .2*np.random.randn(1000, 1)

# Building Xbar, tính theo công thức
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula: ', w_lr.T)

# display result
w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]

x0= np.linspace(0, 1, 2, endpoint= True)
y0 = w_0 + w_1*x0

# draw the fitting bar
plt.plot(X.T, y.T, 'b.') # data
plt.plot(x0, y0, 'y', linewidth = 2) # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()


# tính theo thư viện sklearn
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
w, b = model.coef_[0][0], model.intercept_[0]
sol_sklearn = np.array([b,w])
print('Solution found by sklearn: ',sol_sklearn)


def grad1(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost1(w):
    N = Xbar.shape[0]
    return .5/N * np.linalg.norm(y - Xbar.dot(w)) ** 2
'''
Kiểm tra đạo hàm bằng Numerical Gradient
'''
def numerical_gradient(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n))/ (2*eps)
    return g

def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_gradient(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

print('Checking gradient ...', check_grad(np.random.rand(2,1), cost1, grad1))

def myGd(w_init, grad, eta):
    w = [w_init]
    for i in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:     # Chỗ này chỉ để đảm bảo là trung bình của trị tuyệt đối grad tại từng thành phần là nhỏ hơn 1e-3.
            break                                               # Nếu không chia cho len(w_new) thì với ma trận lớn, mọi thành phần đều nhỏ nhưng tổng của chúng có thể lớn. Vậy nên ta cần tính trung bình.
        w.append(w_new)
    return (w, i)

w_init = np.array([[2], [1]])
(w1 , it1) = myGd(w_init, grad1, 1)
print('Sol found by GD: w = ', w1[-1].T, '\nafter %d iterations.' %(it1+1))

'''
Gradient Descent với Momentum
'''
# hàm 1 biến
# theta_init là điểm khởi tạo ban đầu
def GD_momentum(grad, theta_init, eta, gamma):
    # suppose we want to store history of theta
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for i in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1])
        theta_new = theta[-1] - v_new
        if np.linalg.norm(grad(theta_new)) / np.array(theta_init).size < 1e-3:
            break
        theta.append(theta_new)
        v_old = v_new
    return (theta, i)

# hàm nhiều biến
def GD_momentum2(w_init, grad, eta, gamma):
    w = [w_init]
    v = [np.zeros_like(w_init)]
    for it in range(100):
        v_new = gamma*v[-1] + eta*grad(w[-1])
        w_new = w[-1] - v_new
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
        v.append(v_new)
    return (w, it)

'''
Gradient Descent với NAG 
'''
# Hàm 1 biến
def GD_NAG(grad, theta_init, eta, gamma):
    # suppose we want to store history of theta
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for i in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1] - gamma*v_old)
        theta_new = theta[-1] - v_new
        if np.linalg.norm(grad(theta_new)) / np.array(theta_init).size < 1e-3:
            break
        theta.append(theta_new)
        v_old = v_new
    return (theta, i)

# hàm nhiều biến
def GD_NAG2(w_init, grad, eta, gamma):
    w = [w_init]
    v = [np.zeros_like(w_init)]
    for it in range(100):
        v_new = gamma*v[-1] + eta*grad(w[-1] - gamma*v[-1])
        w_new = w[-1] - v_new
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
        v.append(v_new)
    return (w, it)