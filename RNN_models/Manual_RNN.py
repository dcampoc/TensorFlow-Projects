# -*- coding: utf-8 -*-
"""
Creation of simple 3 neuron RNN layer for checkin the input format of RNNs 
There is no learning here involved, this is just a way to illustrate what the unrolling means for RNNs

@author: dcamp
"""

# RNN has only two batches of data (for time t=0 and t=1). Remember the number of bacthes is different from the size of batches
# Each recurrent neuron has 2 sets of weights (Wx and Wy for the weights of the input X and outpts of X respectively)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# CONSTANTS 

num_inputs = 2
num_neurons = 3
 
# PLACEHOLDERS (in a real dataset this way of setting placeholders and variable won't scale up because it is not feasebel to do this for each timestamp)
x_0 = tf.placeholder(tf.float32,[None,num_inputs])
x_1 = tf.placeholder(tf.float32,[None,num_inputs])

# VARIABLES
# Weights attached to x_0
W_x = tf.Variable(tf.random_normal(shape=[num_inputs,num_neurons]))
# Weights attached to the outout of x 
W_y = tf.Variable(tf.random_normal(shape=[num_neurons,num_neurons]))

b = tf.Variable(tf.zeros([1, num_neurons]))

# GRAPHS (we are repeating the same process a simple NN but now twice because of RNN)
y_0 = tf.tanh(tf.matmul(x_0,W_x) + b)
# It takes the last outputs and multiply them by a set of weights. Then the current input is summed to the result (coding the unrolled RNN)
y_1 = tf.tanh(tf.matmul(y_0,W_y) + tf.matmul(x_1,W_x) + b)

init = tf.global_variables_initializer()

# CREATE DATA 

# timestamp t=0
x0_batch = np.array([[0,1], [2,3], [4,5]])

# timestamp t=1
x1_batch = np.array([[100,101], [102,103], [104,105]])

with tf.Session() as sess:
    sess.run(init)
    
    y0_outputs_vals, y1_outputs_vals = sess.run([y_0, y_1],\
                feed_dict={x_0:x0_batch,x_1:x1_batch})
    

