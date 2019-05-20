from __future__ import print_function
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dir = './data' # path to your data folder
mnist = fetch_mldata('MNIST original',data_home = data_dir)
X = mnist.data
y = mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 10000)

model = LogisticRegression(C = 1e5, solver='lbfgs', multi_class='ovr')  # C is inverse of lam ( C là nghịch đảo của lamda (sử dụng trong weight decay để tránh overfitting, 1 đại lượng))
                                        # tỷ lệ với bình phương norm2 của vector hệ số được cộng vào hàm mất mát để hạn chế độ lớn của hàm số
                                        # solver ='lbfgs là 1 phương pháp tối ưu dựa trên gradient nhưng mạnh hơn GD, multi_class = 'ovr' tương ứng vs one vs rest
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy %.2f %%" %(100 *accuracy_score(y_test, y_pred.tolist())))

'''
Softmax with mini-batch gradient descent
'''