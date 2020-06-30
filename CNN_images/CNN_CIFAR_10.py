# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 23:36:27 2020

@author: dcamp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Put file path as a string here
CIFAR_DIR = 'cifar-10-batches-py/'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict

dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']

all_data = []

# Get directories and a list of 
for direc in dirs:
    all_data.append(unpickle(CIFAR_DIR+direc))

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

print('Classes are shown as follows:'.upper())
# Note that the is a 'b' preceding the names of classes. These are called bytes literals. 

print(batch_meta)

print(data_batch1.keys())

'''
INFO:
data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. 
The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.
'''

X = data_batch1[b'data']
# Take into account that the size of the images is 32 x 32 and there are 3 channels. 32 * 32 * 3 = 3072
# Array of all images reshaped and formatted for viewing
X_1 = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
'''
Explanation of the line above:
X = X.reshape(10000, 3, 32, 32) ---> This line takes all images and organize them into a tensor
X = X.transpose(0,2,3,1) ---> This line keeps the first dimension equal and switches the second dimension to be the last one (for plotting purposes)
X = X.astype('uint8') ---> It casts the information in a type for plotting the images
'''

rand_image = np.random.randint(0,len(X)+1)
plt.imshow(X_1[rand_image])
labels_1 = data_batch1[b'labels'][rand_image]
labels_meta = batch_meta[b'label_names'][labels_1]
print(f'We just printed a: {labels_meta}'.upper())

# This function transform labels into one-hot encoding vectors
def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

class CifarHelper():
    
    def __init__(self):
        self.i = 0
        
        # Grabs a list of all the data batches for training
        self.all_train_batches = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
        # Grabs a list of all the test batches (really just one batch)
        self.test_batch = [test_batch]
        
        # Intialize some empty variables for later on
        self.training_images = None
        self.training_labels = None
        
        self.test_images = None
        self.test_labels = None
    
    # It reshapes and transposes all batches of data (training and testing)
    def set_up_images(self):
        
        print("Setting Up Training Images and Labels")
        
        # Vertically stacks the training images
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)
        
        # Reshapes and normalizes training images
        self.training_images = self.training_images.reshape(train_len,3,32,32).transpose(0,2,3,1)/255
        # One hot Encodes the training labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)
        
        print("Setting Up Test Images and Labels")
        
        # Vertically stacks the test images
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)
        
        # Reshapes and normalizes test images
        self.test_images = self.test_images.reshape(test_len,3,32,32).transpose(0,2,3,1)/255
        # One hot Encodes the test labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    # Batch size is assumed to be 100 (images come in groups of 100)
    def next_batch(self, batch_size):
        # Note that the 100 dimension in the reshape call is set by an assumed batch size of 100
        x = self.training_images[self.i:self.i+batch_size].reshape(100,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y


# Before Your tf.Session run these two lines
ch = CifarHelper()
ch.set_up_images()

# During your session to grab the next batch use this line
# (Just like we did for mnist.train.next_batch)
# batch = ch.next_batch(100)

x = tf.placeholder(tf.float32, shape=[None,32,32,3])
y_true = tf.placeholder(tf.float32, shape=[None,10])

# No need for shape here, it is just a number carrying the probability of performing dropout
hold_prob = tf.placeholder(tf.float32)


# INIT_WEIGHTS 

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(init_random_dist)

# INIT BIAS 

def init_bias(shape):
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)

# CONV2D (it creates a 2D convolution)

def conv2d(x,W):
    # x ---> Actual input tensor [batch, H, W, Channels]
    # W ---> Kernel [filter H, filter W, Channels IN, Channels OUT]
    # padding='SAME' means zero padding
    
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    
# POOLING
def max_pool_2by2(x):
    # x ---> Actual input tensor [batch, H, W, Channels]
    # The pooling is done over H: high and W:width
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# CONVOLUTIONAL LAYER (it passes the result of the convolutional layer to an activation function)
def convolutional_layer(input_x,shape):
    W = init_weights(shape)
    # Biases go along the third dimension 
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W) + b)

# NORMAL FULLY CONNECTED 
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])    
    return tf.matmul(input_layer,W) + b

# first two sizes is up to you (size of filters) but the 3rd should be 3, because of the number of channels. The 4th index is also up to us, we consider 32 filters 
convo_1 = convolutional_layer(x,shape=[4,4,3,32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[4,4,32,64])
convo_2_pooling = max_pool_2by2(convo_2)

# reshaping into an 8 by 8 image 
convo_2_flat = tf.reshape(convo_2_pooling,[-1,8*8*64])

full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

# Number of labels: 10
y_pred = normal_full_layer(full_one_dropout,10)

# LOSS FUNCTION 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

# OPTIMIZER 
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

steps = 4000

with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(steps):
        # Using the helper function, the bacthes are extracted form the dataset by using the following function 
        batch = ch.next_batch(100)
        
        # During training each neuron has 50% of being held 
        sess.run(train,feed_dict={x:batch[0],y_true:batch[1],hold_prob:0.5})
        
        if i%100 == 0:
            print(f'ON STEP: {i}')
            
            matches = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            
            # All neurons should be held
            final_acc = sess.run(acc,feed_dict={x:ch.test_images,y_true:ch.test_labels,hold_prob:1.0}) 
            print(f'ACCURACY: {final_acc}')
            print('\n')
            


