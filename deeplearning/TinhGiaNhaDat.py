from __future__ import print_function
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
# load data from scv

data = pd.read_csv('./data/data_linear.csv').values

N = data.shape[0]

X = data[:, 0].reshape(-1, 1) # dua ve ma tran cot, moi hang la 1 diem du lieu
y = data[:, 1].reshape(-1, 1)



# building Xbar (them gia tri 1 vao)
ones = np.ones((N, 1))
Xbar = np.concatenate((ones, X), axis = 1)

A = Xbar.T.dot(Xbar)
b = Xbar.T.dot(y)

w = np.dot(np.linalg.pinv(A), b)

w0 = w[0]
w1 = w[1]

x0 = np.linspace(30, 100, 2)
y0 = w0 + w1*x0

# Biểu đồ dữ liệu
plt.plot(X, y, 'ro')
plt.plot(x0, y0)
plt.xlabel('mét vuông')
plt.ylabel('giá')
plt.show()

# du doan gia nha 50m2
x1 = 50
y1 = w0 + w1*x1
print('50m2 co gia la: %.2f' %y1[0])