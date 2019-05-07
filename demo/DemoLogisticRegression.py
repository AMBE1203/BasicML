from __future__ import print_function
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_pdf import PdfPages
from display_network import *

mnist = fetch_mldata('MNIST original', data_home = './data')
N, d = mnist.data.shape

X_all = mnist.data
y_all = mnist.target

X0 = X_all[np.where(y_all == 0)[0]] # all digit 0
X1 = X_all[np.where(y_all == 1)[0]] # all digit 1
y0 = np.zeros(X0.shape[0]) # class 0 label
y1 = np.ones(X1.shape[0]) # class 1 label

X = np.concatenate((X0, X1), axis = 0) # all data
y = np.concatenate((y0, y1)) # all label

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2000)
model = LogisticRegression(C = 1e5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy %.2f %%" %(100 * accuracy_score(y_test, y_pred.tolist())))

# Tìm ảnh bị phân loại lỗi
mis = np.where((y_pred - y_test) != 0)[0]
Xmis = X_test[mis, :]

filename = 'mnist_mis.png'
with PdfPages(filename) as pdf:
    plt.axis('off')
    A = display_network(Xmis.T, 1, Xmis.shape[0])
    f2 = plt.imshow(A, interpolation = 'nearest')
    plt.gray()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()