'''
K-means clustering
Đầu vào: Ma trận dữ liệu X ∈ R(d,N)và số lượng cluster cần tìm K < N.
Đầu ra: Ma trận các centroid M ∈ R(d,K) và ma trận label Y ∈ R(N,K) .
1. Chọn K điểm bất kỳ trong training set làm các centroid ban đầu.
2. Phân mỗi điểm dữ liệu vào cluster có centroid gần nó nhất.
3. Nếu việc phân nhóm dữ liệu vào từng cluster ở bước 2 không thay đổi so với vòng
lặp trước nó thì ta dừng thuật toán.
4. Cập nhật centroid cho từng cluster bằng cách lấy trung bình cộng của tất các các
điểm dữ liệu đã được gán vào cluster đó sau bước 2.
5. Quay lại bước 2.
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt # hiển thị dữ liệu
from scipy.spatial.distance import cdist # tính khoảng cách giữa các cặp điểm trong 2 tập hợp
import random
from sklearn.cluster.tests.test_affinity_propagation import n_clusters
np.random.seed(18)

'''
Dữ liệu được tạo bằng cách lấy ngẫu nhiên 500 điểm cho mỗi cluster theo phân phối chuẩn
có kỳ vọng lần lượt là (2, 2), (8, 3) và (3, 6) , ma trận hiệp phương sai giống nhau và là
ma trận đơn vị. Mỗi điểm dữ liệu là 1 hàng của ma trận dữ liệu.
'''
means = [[2,2],[8,3],[3,6]]

cov = [[1,0],[0,1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0,X1,X2), axis = 0) # Nối 3 ma trận dữ liệu theo chiều y
K = 3 # number of cluster
original_label = np.asarray([0]*N + [1]*N + [2]*N).T

# hiển thị dữ liệu
def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :] # chọn các điểm dữ liệu của X ở các hàng mà có label == 0
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4 , alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4 , alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4 , alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()

# khởi tạo các centroids ban đầu
def kmeans_init_centroids(X,k):
    return X[np.random.choice(X.shape[0], k, replace = False)] # randomly pick k rows of X as initial centroids

# tìm label mới cho các điểm khi cố định centroids
def kmeans_assign_labels(X, centroids):
    D = cdist(X, centroids)  # tính toán khoảng cách giữa các điểm dữ liệu với centers
    return np.argmin(D, axis=1) # trả về vị trí của center gần nhất

# cập nhật các centroid khi biết label của mỗi điểm dữ liệu
def kmeans_update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk= X[labels == k,:] # chọn tất cả các điểm có nhãn ==k
        centroids[k,:] = np.mean(Xk, axis=0) # lấy trung bình của tất cả các điểm trong cluster thứ k
    return centroids

# kiểm tra điều kiện dừng của thuật toán
def has_converged(centroids, new_centroids):
    return (set([tuple(a) for a in centroids]) == 
     set([tuple(a) for a in new_centroids]))  # return true if two sets of centroids are the same

# KMeans
def kmeans(X, K):
    centroids = [kmeans_init_centroids(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centroids[-1]))
        new_centroids = kmeans_update_centroids(X, labels[-1], K)
        if has_converged(centroids[-1], new_centroids):
            break
        centroids.append(new_centroids)
        it += 1
    return (centroids,labels,it)

(centroids, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:\n', centroids[-1])
print('Count loop: ',it)
kmeans_display(X, labels[-1])

# Demo with scikit-learn

from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3, random_state= 0).fit(X)
print('Centers found by scikit-learn:')
print(model.cluster_centers_)
pred_label = model.predict(X)
kmeans_display(X, pred_label)