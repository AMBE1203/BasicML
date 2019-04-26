from __future__ import print_function
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster.tests.test_affinity_propagation import n_clusters


# Phân loại chữ số viết tay
data_dir = './data' # path to your data folder
mnist = fetch_mldata('MNIST original', data_home=data_dir)
print("Shape of mnist data:", mnist.data.shape)

K = 10 # Number of cluster
N = 10000
X = mnist.data[np.random.choice(mnist.data.shape[0], N)]
kmeans = KMeans(n_clusters = K).fit(X)
pred_label = kmeans.predict(X)  # dự đoán label của dữ liệu, các centroids được lưu trong biến kmeans.cluster_centers_



# tách vậy thể trong ảnh
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
  
img = mpimg.imread('./data/girl_kmeans.jpg')
plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis('off')
#plt.show()
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2])) # biến đổi bức ảnh thành 1 mà trận mà mỗi hàng là 1 pixel với 3 giá trị màu
for K in [3]:
    kmeans = KMeans(n_clusters= K).fit(X)
    label = kmeans.predict(X)
    img2 = np.zeros_like(X)
    # replace each pixel by its center
    for k in range(K):
        img2[label == k] = kmeans.cluster_centers_[k]
    # reshape and display output image
    img3 = img2.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img3, interpolation='nearest')
    plt.axis('off')
    plt.show()