from __future__ import print_function
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split # for splitting data (để chia dữ liệu)
from sklearn.metrics import accuracy_score # for evaluating result (để đánh giá kết quả)

# load 130 mẫu làm test set, 20 mẫu làm trainning set
np.random.seed(7) # với các biến số trong seed khác nhau thì cho ra các số ngẫu nhiên khác nhau, nếu k đổi biến số trong seed thì chỉ ra 1 số ngẫu nhiên
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

print('Label: ', np.unique(iris_Y)) # hàm np.unique trả về các phần tử có trong 1 tập hợp, 1 ma trận, 1 mảng 

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size=130)
print('Train size:', X_train.shape[0], ', Test size:', X_test.shape[0])

# TH1 K= 1
model = neighbors.KNeighborsClassifier(n_neighbors=1, p = 2) # K = 1, p = 2 <--> sử dụng norm2 để tính khoảng cách
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy of 1KNN: %.2f %%' %(100* accuracy_score(y_test,y_pred)))
print("Print results for 20 test data points:")
print("Predicted labels: ", y_pred[20:40])
print("Ground truth    : ", y_test[20:40])

# TH2 K=7
model = neighbors.KNeighborsClassifier(n_neighbors=7, p = 2) 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy of 7KNN: %.2f %%' %(100* accuracy_score(y_test,y_pred)))

'''Trong kỹ thuật major voting bên trên, mỗi trong bảy điểm gần nhất được coi là có vai trò
như nhau và giá trị lá phiếu của mỗi điểm này là như nhau. Như thế có thể không công
bằng, vì những điểm gần hơn cần có trọng số cao hơn. Vì vậy, ta sẽ đánh trọng số khác
nhau cho mỗi trong bảy điểm gần nhất này. Cách đánh trọng số phải thoải mãn điều kiện là
một điểm càng gần điểm kiểm thử phải được đánh trọng số càng cao. Cách đơn giản nhất
là lấy nghịch đảo của khoảng cách này. Trong trường hợp test data trùng với một điểm dữ
liệu trong training data, tức khoảng cách bằng 0, ta lấy luôn đầu ra của điểm training data
Scikit-learn giúp chúng ta đơn giản hóa việc này bằng cách gán thuộc tính weights = ’
distance’ . (Giá trị mặc định của weights là ’uniform’ , tương ứng với việc coi tất cả các điểm
lân cận có giá trị như nhau như ở trên).
'''

# TH3 KNN Thêm trọng số vào mỗi điểm
model = neighbors.KNeighborsClassifier(n_neighbors=7, p = 2, weights='distance') 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy of 7KNN( 1/ distance weights): %.2f %%' %(100* accuracy_score(y_test,y_pred)))

# TH4 KNN với trọng số tự định nghĩa

'''
wi = exp( - norm2( z - xi) ^ 2 / sigma ^ 2)
wi là trọng số của điểm gần thứ i (xi của điểm đang xét z), sigma là 1 số dương
'''
def myWeight(distances):
    sigma2 = .4 # we can change this number
    return np.exp(-distances ** 2/sigma2)

model = neighbors.KNeighborsClassifier(n_neighbors=7, p = 2, weights=myWeight) 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy of 7KNN(custom weights): %.2f %%' %(100* accuracy_score(y_test,y_pred)))

