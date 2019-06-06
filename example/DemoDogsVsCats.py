from __future__ import print_function
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from PIL import Image


train_directory = './data/cat_dog/train/'
test_directory = './data/cat_dog/test/'

def images(image_directory):
    return [image_directory+image for image in os.listdir(image_directory)]

train_image_names = images(train_directory)



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

Y_data = extract_labels(train_image_names)

print(Y_data)

'''
image_size = [ ]

for i in train_image_names: # images(file_directory)
    im = Image.open(i)
    image_size.append(im.size) # A list with tuples: [(x, y), …] 

# Get mean of image size (Ref: stackoverflow)
mean = [sum(y) / len(y) for y in zip(*image_size)]
print(mean)
'''

'''
STANDARD_SIZE = (400, 350)
# Function to read image, change image size and transform image to matrix
def img_to_matrix(filename, verbose=False):
        
    
 #   takes a filename and turns it into a numpy array of RGB pixels
    
    img = Image.open(filename)
    # img = Image.fromarray(filename)
    if verbose == True:
        print( "Changing size from %s to %s") % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

# Function to flatten numpy array
def flatten_image(img):
    
  #  takes in an (m, n) numpy array and flattens it   into an array of shape (1, m * n)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

## Prepare training data
data = []
for i in images(train_directory):
    img = img_to_matrix(i)
    img = flatten_image(img)
    data.append(img)
    
data = np.array(data)

print(data.shape)
'''


