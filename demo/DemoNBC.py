from __future__ import print_function
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from scipy.sparse import coo_matrix # for sparse matrix (ma trận thưa có số phần tử == 0 nhiều hơn số phần tử != 0)
from sklearn.metrics import accuracy_score # for evaluating result

# data path and file name
path = './data/ex6DataPrepared/'
train_data_fn = 'train-features.txt'
test_data_fn = 'test-features.txt'
train_label_fn = 'train-labels.txt'
test_label_fn = 'test-labels.txt'

nwords = 2500

# đọc dữ liệu từ file data_fn vs label tương ứng được lưu trong file label_fn
def read_data(data_fn, label_fn):
    ## read label_fn
    with open(path+label_fn) as f:
        content = f.readlines()
    label = [int(x.strip()) for x in content]  # x.trip() xóa khoảng trắng ở đầu và cuối của từng dòng

    ## read data_fn
    with open(path+data_fn) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    dat = np.zeros((len(content), 3), dtype = int)
    for i, line in enumerate(content):
        a = line.split(' ')
        dat[i,:] = np.array([int(a[0]), int(a[1]), int(a[2])])
    
    data = coo_matrix((dat[:,2], (dat[:,0]-1, dat[:,1]-1)), shape = (len(label), nwords))
    return (data, label)

(train_data, train_label) = read_data(train_data_fn, train_label_fn)
(test_data, test_label) = read_data(test_data_fn, test_label_fn)
clf = MultinomialNB()
clf.fit(train_data, train_label)
y_pred = clf.predict(test_data)
print('Training size = %d, accuracy = %.2f%%' %(train_data.shape[0],accuracy_score(test_label, y_pred)*100))