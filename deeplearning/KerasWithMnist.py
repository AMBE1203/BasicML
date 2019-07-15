from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# load data from mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_val, y_val = X_train[50000:60000, :], y_train[50000:60000]
X_train, y_train =X_train[:50000, :], y_train[:50000]


# reshape data
X_train = X_train.reshape(X_train.shape[0], 28, 28 , 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# one-hot-coding
Y_train = np_utils.to_categorical(y_train, 10)
Y_val = np_utils.to_categorical(y_val, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print('Y before one-hot-coding ', y_train[0])
print('Y after one-hot-coding ', Y_train[0])

# Build model
'''
Model = Sequential() để nói cho keras là ta sẽ xếp các layer lên nhau để tạo model.
Ví dụ input -> CONV -> POOL -> CONV -> POOL -> FLATTEN -> FC -> OUTPUT
Ở layer đầu tiên cần chỉ rõ input_shape của ảnh, input_shape = (W, H, D), ta dùng ảnh xám kích thước (28,28) nên input_shape = (28, 28, 1)
Khi thêm Convolutional Layer ta cần chỉ rõ các tham số: K (số lượng layer), kernel size (W, H), hàm activation sử dụng. 
cấu trúc: model.add(Conv2D(K, (W, H), activation='tên_hàm_activation'))
Khi thêm Maxpooling Layer cần chỉ rõ size của kernel, model.add(MaxPooling2D(pool_size=(W, H)))
Bước Flatten chuyển từ tensor sang vector chỉ cần thêm flatten layer.
Để thêm Fully Connected Layer (FC) cần chỉ rõ số lượng node trong layer và hàm activation sử dụng trong layer,
cấu trúc: model.add(Dense(số_lượng_node activation='tên_hàm activation'))
'''

model = Sequential()
# add convolutional layer with 32 kernel, size of kernel 3x3
# sigmoid is activation, first layer with input shape
model.add(Conv2D(32, (3, 3), activation = 'sigmoid', input_shape = (28, 28, 1)))

# add convolution layer
model.add(Conv2D(32, (3, 3), activation = 'sigmoid'))

# add max pooling layer
model.add(MaxPooling2D(pool_size = (2, 2)))

# using flatten layer for convert tensor to vector
model.add(Flatten())

# add fully connected layer with 128 nodes and sigmoid
model.add(Dense(128, activation = 'sigmoid'))

# output layer with 10 nodes and softmax funtion
model.add(Dense(10, activation = 'softmax'))

# compile model, chỉ rõ hàm loss_function nào được sử dụng, phương thức 
# dùng để tối ưu hàm loss function.

model.compile(loss = 'categorical_crossentropy', optimizer ='adam', metrics=['accuracy'])

# train model
H = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=32, epochs=10, verbose=1)

# vẽ đồ thị loss, accuracy của traning set và validation set
fig = plt.figure()
numOfEpoch = 10
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()

# đánh giá model với dữ liệu test set
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

# predict
plt.imshow(X_test[0].reshape(28,28), cmap='gray')

y_predict = model.predict(X_test[0].reshape(1,28,28,1))
print('Giá trị dự đoán: ', np.argmax(y_predict))
