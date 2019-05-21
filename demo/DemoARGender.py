from __future__ import print_function
import numpy as np
from sklearn import linear_model # for logistic regression
from sklearn.metrics import accuracy_score # for evaluation
from scipy import misc # for loading image
import matplotlib.pyplot as plt
np.random.seed(12)

# phân chia các trainning set và test set, lựa chọn các view
path = './data/cropped_faces/'
train_ids = np.arange(1, 4)
test_ids = np.arange(4, 7)
view_ids = np.arange(1, 15)

D = 120 * 165 # original dimension
d = 500 # new dimension

ProjectionMatrix = np.random.randn(D, d) 

# xây dựng danh sách các tên file
def build_list_fn(pre, img_ids, view_ids):
    '''
    Input:
        pre = 'M-' or 'W-'
        img_ids = indexes of images
        view_ids = indexes of views
    Output:
        a list of filenames
    '''
    list_fn = []
    for im_id in img_ids:
        for v_id in view_ids:
            fn = path + pre + str(im_id).zfill(2) + '-' + \
                str(v_id).zfill(2) + '.bmp'
            list_fn.append(fn)
    return list_fn

# feature extraction xây dựng dữ liệu cho trainning set và test set

def rgb2gray(rgb):
    # Y' = 0.299R + 0.587G + 0.114B
    return rgb[:,:,0]*.299 + rgb[:,:,1]*.587 + rgb[:,:,2]*.144

# feature extraction
def vectorize_img(filename):
    #load image
    rgb = misc.imread(filename)
    # convert to gray scale
    gray = rgb2gray(rgb)
    # vectorizarion each row is a data point
    im_vec = gray.reshape(1, D)
    return im_vec

def build_data_matrix(img_ids, view_ids):
    total_imgs = img_ids.shape[0]*view_ids.shape[0]*2

    X_full = np.zeros((total_imgs, D))
    y = np.hstack((np.zeros((total_imgs//2, )), np.ones((total_imgs//2, ))))

    list_fn_m = build_list_fn('m-', img_ids, view_ids)
    list_fn_w = build_list_fn('w-', img_ids, view_ids)
    list_fn = list_fn_m + list_fn_w
    
    for i in range(len(list_fn)):
        X_full[i, :] = vectorize_img(list_fn[i])

    X = np.dot(X_full, ProjectionMatrix)

    return (X, y)

(X_train_full, y_train) = build_data_matrix(train_ids, view_ids)
x_mean = X_train_full.mean(axis = 0)
x_var  = X_train_full.var(axis = 0)

def feature_extraction(X):
    return (X - x_mean)/x_var     

X_train = feature_extraction(X_train_full)
X_train_full = None ## free this variable 

(X_test_full, y_test) = build_data_matrix(test_ids, view_ids)
X_test = feature_extraction(X_test_full)
X_test_full = None 

logreg = linear_model.LogisticRegression(C = 1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))


# sử dụng hàm predict_proba() để dự đoán giới tính của 1 ảnh bất kỳ

def feature_extraction_fn(fn):
    im = vectorize_img(fn)
    im1 = np.dot(im, ProjectionMatrix)
    return feature_extraction(im1)

fn1 = path + 'm-07-10.bmp'
fn2 = path + 'w-06-01.bmp'
fn3 = path + 'm-05-14.bmp'
fn4 = path + 'w-07-02.bmp'

x1 = feature_extraction_fn(fn1)
p1 = logreg.predict_proba(x1)
print(p1)

x2 = feature_extraction_fn(fn2)
p2 = logreg.predict_proba(x2)
print(p2)

x3 = feature_extraction_fn(fn3)
p3 = logreg.predict_proba(x3)
print(p3)

x4 = feature_extraction_fn(fn4)
p4 = logreg.predict_proba(x4)
print(p4)


# hiển thị lên màn hình
def display_result(fn):
    x1 = feature_extraction_fn(fn)
    p1 = logreg.predict_proba(x1)
    print(logreg.predict_proba(x1))
    rgb = misc.imread(fn)
    
    
    fig = plt.figure()
#     gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
#     plt.subplot(1, 2, 1)
    plt.figure(facecolor="white")
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(rgb)
#     plt.axis('off')
#     plt.show()
    plt.subplot(122)
    plt.barh([0, 1], p1[0], align='center', alpha=0.9)
    plt.yticks([0, 1], ('man', 'woman'))
    plt.xlim([0,1])
    plt.show()
    
    
   
    # load an img 
fn1 = path + 'm-07-10.bmp'
fn2 = path + 'w-05-01.bmp'
fn3 = path + 'm-05-14.bmp'
fn4 = path + 'w-07-02.bmp'

display_result(fn1)
display_result(fn2)
display_result(fn3)
display_result(fn4)