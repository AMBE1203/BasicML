'''
Với KNN, trong bài toán classification, nhãn của một điểm dữ liệu mới được suy ra trực tiếp
từ K điểm dữ liệu gần nhất trong tập huấn luyện. Nhãn đó có thể được quyết định bằng
bầu chọn theo đa số (major voting) trong số K điểm gần nhất, hoặc nó có thể được suy ra
bằng cách đánh trọng số khác nhau cho mỗi trong các điểm gần nhất đó rồi suy ra kết quả.
Chi tiết sẽ được nêu trong phần tiếp theo. Trong bài toán regresssion, đầu ra của một điểm
dữ liệu sẽ bằng chính đầu ra của điểm dữ liệu đã biết gần nhất (trong trường hợp K = 1),
hoặc là trung bình có trọng số của đầu ra của những điểm gần nhất, hoặc bằng một mối
quan hệ dựa trên các điểm gần nhất đó và khoảng cách tới chúng.
Một cách ngắn gọn, KNN là thuật toán đi tìm đầu ra của một điểm dữ liệu mới bằng cách
chỉ dựa trên thông tin của K điểm dữ liệu gần nhất trong tập huấn luyện.
'''
from __future__ import print_function
import numpy as np
from time import time


M = 100
d, N = 1000, 10000 # dimension, number of trainning points
X = np.random.randn(N, d) # N - d dimension points
z = np.random.randn(d)
Z = np.random.randn(M, d)


# tính bình phương khoảng cách Euclid giữa 2 vector z và x. Tính hiệu rồi lấy bình phương norm của vector hiệu
def dist_pp(z, x):
    d = z - x.reshape(z.shape) # z và x phải có cùng chiều
    return np.sum(d*d)

# tính bình phương khoảng cách giữa z và mỗi hàng của X
def dist_ps_naive(z, X):
    N = X.shape[0]
    res = np.zeros((1,N))
    for i in range(N):
        res[0][i] = dist_pp(z,X[i])
    return res

# tính bình phương khoảng cách giữa z và mỗi hàng của X. Khi có nhiều điểm dữ liệu được lưu trong X, thì ta tính tổng
# bình phương của mỗi X[i] và tính tích X.T * z
def dist_ps_fast(z, X):
    X2 = np.sum(X*X, 1) # ma trận vuông của norm2 của mỗi hàng của X
    z2 = np.sum(z*z)
    return X2 + z2 - 2 * np.dot(X,z)


t1 = time()
D1 = dist_ps_naive(z, X)
print('naive point2set, running time:', time() - t1, 's')
t1 = time()
D2 = dist_ps_fast(z, X)
print('fast point2set , running time:', time() - t1, 's')
print('Result difference:', np.linalg.norm(D1 - D2))



# tính khoảng cách từ mỗi điểm của tập Z đến mỗi điểm của tập Z
def dist_ss_0(Z, X):
    M = Z.shape[0]
    N = X.shape[0]
    res = np.zeros((M, N))
    for i in range(M):
        res[i] = dist_ps_fast(Z[i], X)
    return res

def dist_ss_fast(Z, X):
    X2 = np.sum(X*X,1)
    Z2 = np.sum(Z*Z,1)
    return Z2.reshape(-1,1) + X2.reshape(1, -1) - 2*Z.dot(X.T)



t1 = time()
D3 = dist_ss_0(Z, X)
print('half fast set2set running time:', time() - t1, 's')
t1 = time()
D4 = dist_ss_fast(Z, X)
print('fast set2set running time:', time() - t1, 's')
print('Result difference:', np.linalg.norm(D3 - D4))
    

