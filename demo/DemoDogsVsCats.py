from __future__ import print_function
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from PIL import Image
import os
from scipy import misc # for loading image


data_dir  = './data/datachomoe/train/'
test_dir = './data/datachomoe/test/'
D = 500

def rgb2gray(rgb):
    # Y' = 0.299R + 0.587G + 0.114B
    return rgb[:,:,0]*.299 + rgb[:,:,1]*.587 + rgb[:,:,2]*.144

# feature extraction
def vectorize_img(filename):
    #load image
   # rgb = misc.imread(filename)
    # convert to gray scale
    gray = rgb2gray(filename)
    print(gray.shape)
    # vectorizarion each row is a data point
    im_vec = gray.reshape(1, D)
    print(im_vec.shape)
    return im_vec

# Resize image
def resize_image(img, size):
    """
    input: 
        img = PIL image
        size = side length of square image (output)
    """
    # Resizing image
    x, y = img.size
    if x > y:
        nx = size
        ny = int(size * y/x + 0.5)
    else:
        nx = int(size * x/y + 0.5)
        ny = size
    temp_res = img.resize((nx, ny), resample=Image.ANTIALIAS)
    
    # Padding borders to create a square image
    temp_pad = Image.new('RGB', (size, size), (128, 128, 128))
    temp = ((size - nx)//2, (size - ny)//2)
    temp_pad.paste(temp_res, temp)

    return temp_pad

SIZE = 96
CHANNELS = 3


# Image to Numpy Array
def image_to_array(img):
    try:

        arr = np.asarray(img, dtype='uint8')
        xxx = arr.reshape(1, SIZE * SIZE * CHANNELS)

    except SystemError:
        arr = np.asarray(img, dtype='uint8')
        xxx = arr.reshape(1, SIZE * SIZE * CHANNELS)

    return xxx

dog_images = [data_dir + img for img in os.listdir(data_dir) if 'dog' in img]
cat_images = [data_dir + img for img in os.listdir(data_dir) if 'cat' in img]

images = dog_images + cat_images
np.random.shuffle(images)


from sklearn.decomposition import PCA


def prepare_data(images):
    N = len(images)
    print(N)
    # Create an ndarray (N, C, H, W)
    data = np.zeros((N , CHANNELS* SIZE *SIZE))
    print(data.shape)
    # Resize, reshape and append
    for i, img in enumerate(images):
        img_ = Image.open(img)
        temp_resize = resize_image(img_, SIZE)
        temp_array = image_to_array(temp_resize)
        data[i, :] = temp_array
        if i % 2500 == 0: 
            print ('Processed {} of {} images'.format(i, N))
           
    return data  

X_all = prepare_data(images)


# Generate labels
# Function to extract labels
def extract_labels(file_names):
    '''Create labels from file names: Cat = 0 and Dog = 1'''
    
    # Create empty vector of length = no. of files, filled with zeros 
    n = len(file_names)
    y = np.zeros(n, dtype = np.int32)
    
    # Enumerate gives index
    for i, filename in enumerate(file_names):
        
        # If 'cat' string is in file name assign '0'
        if 'cat' in str(filename):
            y[i] = 0
        else:
            y[i] = 1
    return y
y_all = extract_labels(images)



# # Split data into training and testing

y_train_full = y_all[:1000]
y_test_full = y_all[1000:]

print(y_train_full)

X_train_full = X_all[:1000]
X_test_full = X_all[1000:]



x_mean = X_train_full.mean(axis = 0)
x_var = X_train_full.var(axis = 0)



X_train = (X_train_full - x_mean)/x_var

X_test = (X_test_full - x_mean)/x_var


logreg = linear_model.LogisticRegression(C = 1e5)
logreg.fit(X_train, y_train_full)
y_pred = logreg.predict(X_test)
print("Accuracy: %.2f %%" %(100*accuracy_score(y_test_full, y_pred)))



def feature_extraction_fn(img):
    img_ = Image.open(img)
    temp_resize = resize_image(img_, SIZE)
    temp_array = image_to_array(temp_resize)
    return temp_array


# hiển thị lên màn hình
def display_result(fn):
    x1 = feature_extraction_fn(fn)
    p1 = logreg.predict_proba(x1)
    print(logreg.predict_proba(x1))
    rgb = misc.imread(fn)
    
    
    fig = plt.figure()
    plt.figure(facecolor="white")
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(rgb)
    plt.subplot(122)
    plt.barh([0, 1], p1[0], align='center', alpha=0.9)
    plt.yticks([0, 1], ('cat', 'dog'))
    plt.xlim([0,1])
    plt.show()


fn1 = test_dir + '101.jpg'
fn2 = test_dir + '31.jpg'
fn3 = test_dir + '12.jpg'
fn4 = test_dir + '27.jpg'



display_result(fn1)
display_result(fn2)
display_result(fn3)
display_result(fn4)

