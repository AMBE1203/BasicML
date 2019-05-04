from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as cdist
np.random.seed(12)


# có 2 lớp, mỗi lớp 10 phần tử, lấy ngẫu nhiên theo phân phối chuẩn có ma trận hiệp phương sai là cov và vector kỳ vọng được lưu trong means
# sau đó, tạo dữ liệu mở rộng bằng cách thêm 1 vào đầu mỗi điểm dữ liệu
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
Y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # chứa nhãn của mỗi điểm dữ liệu trong X
# Xbar
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)  # chứa dữ liệu X mở rộng, thêm 1 đặc trưng = 1 vào đầu vector đặc trưng

# dự đoán nhãn cho mỗi cột của X cho w
# X là 1 mảng 2d có shape(N, d)
# W là 1 mảng 1d có shape (d, )
def predict(w, X):
    return np.sign(np.dot(w.T, X)) # np.sign là hàm xác định dấu

# kiểm tra điều kiện dừng. chỉ cần so sánh predict(w, X) với ground truth Y. nếu giống nhau thì dừng thuật toán
def has_converged(X, Y, w):
    return np.array_equal(predict(w, X), Y)

def perceptron(X, Y, w_init):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []
    while True:
        # mix data
        mix_id = np.random.permutation(N)  # Trả về 1 array chứa các index của data đã xáo trộn
        for it in range(N):
            xi = X[:, mix_id[it]].reshape(d, 1) # lấy ra giá trị 1 data thứ it trong array X sau đó reshape thành ndarray 3x1
            yi = Y[0, mix_id[it]] # lấy ra nhãn của data thứ it trong array trên
            if predict(w[-1], xi)[0] != yi: # misclassified point
                mis_points.append(mix_id[it])
                w_new = w[-1] + yi*xi
                w.append(w_new)
        if has_converged(X, Y, w[-1]):
            break
    return (w, mis_points)

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = perceptron(X, Y, w_init)
print(w)
