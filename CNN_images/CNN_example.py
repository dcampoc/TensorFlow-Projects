# -*- coding: utf-8 -*-
"""
CNN applied to MNIST data

@author: dcamp
"""
# Note: If it appears an error of the type: "ResourceExhaustedError: OOM when allocating tensor...", reduce the batch size or restart the kernel (ctrl + w on spyder)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# If the following line is not working, please copy and paste the missing folders you find the directory tensorflow/tensorflow/examples/ of the following link: https://github.com/tensorflow/tensorflow 
# You sould copy and paste those folders in your own ...\Python3(or name of your env)\Lib\site-packages\tensorflow\examples directory. 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNISTdata/",one_hot=True)

type(mnist)

# HELPER 

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

# PLACEHOLDER
x = tf.placeholder(tf.float32, shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])

# LAYERS 
# the first layer correspond to the image, which is restored to its original size (28,28), it has only one channel because it is gray-scale
x_image = tf.reshape(x,[-1,28,28,1])
# this convolution computes 32 features for each 5 by 5 patch. The third dimension is the number of channels: one (gray-scale). Output channels = 32 
convo_1 = convolutional_layer(x_image,shape=[5,5,1,32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[5,5,32,64])
convo_2_pooling = max_pool_2by2(convo_2)

# 7 by 7 image 
convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
# We decide we want 1024 neurons for performing the classification task
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

# DROPOUT 
hold_prob = tf.placeholder(tf.float32)
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
        # Batches of 50
        batch_x, batch_y = mnist.train.next_batch(batch_size=50)
        # During training each neuron has 50% of being held 
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        
        if i%100 == 0:
            print(f'ON STEP: {i}')
            
            matches = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            
            # All neurons should be held
            final_acc = sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0}) 
            print(f'ACCURACY: {final_acc}')
            print('\n')
            