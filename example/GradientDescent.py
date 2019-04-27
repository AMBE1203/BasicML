from __future__ import print_function
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.pyplot as plt


def grad(x):
    return 2*x + 5*np.cos(x)

def cost(x):
    return x ** 2 + 5*np.sin(x)

def myGD1(x0, eta):
    x = [x0]
    print(x)
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
model.fit(X.reshape(-1,1), y.reshape(-1,1))
w, b = model.coef_[0][0], model.intercept_[0]
sol_sklearn = np.array([b,w])
print('Solution found by sklearn: ',sol_sklearn)


