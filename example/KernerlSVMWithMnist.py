from __future__ import print_function
import numpy as np
from sklearn import svm
from sklearn.datasets import fetch_mldata


data_dir = './data' # path to your data folder
mnist = fetch_mldata('MNIST original',data_home = data_dir)
X_all = mnist.data/255
y_all = mnist.target

digits = [0, 1, 2, 3]
ids = []
for d in digits:
    ids.append(np.where(y_all == d)[0])

selected_ids = np.concatenate(ids, axis = 0)
X = X_all[selected_ids]
y = y_all[selected_ids]

print('Number of samples = ', X.shape[0])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 24000)
model = svm.SVC(kernel = 'rbf', gamma=.1, coef0=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy: %.2f %%' %(100%accuracy_score(y_test, y_pred)))


