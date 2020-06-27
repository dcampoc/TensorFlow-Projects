# -*- coding: utf-8 -*-
"""
Softmax regression approach for classification on MNIST using TF
(Vanilla NN)

@author: dcamp
"""

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

print(f'number of training images: {mnist.train.num_examples}')
print(f'number of testing images: {mnist.test.num_examples}')
print('\n')

original_single_data = mnist.train.images[0] 
print(f'Note that the shape of a single data (image) is: {original_single_data.shape}')
print('It is already a flatten vector!'.upper())
print('\n')
# Converting the flatten vector into the original image
single_image = mnist.train.images[0].reshape(28,28)

plt.imshow(single_image, cmap='gist_gray')

print('Note that the data is already normalized'.upper())
print(f'minimum image value: {single_image.min()}')
print(f'maximum image value: {single_image.max()}')
print('\n')

# PLACEHOLDERS
x = tf.placeholder(tf.float32,shape=[None,784])

# VARIABLES (initialize as zeros is not a good idea but in this example we do it)
W = tf.Variable(tf.zeros([784,10,]))
b = tf.Variable(tf.zeros([10]))

# CREATE GRAPH OPERATIONS
y = tf.matmul(x,W) + b

# LOSS FUNCTION 
y_true = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# -OPTIMIZER
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# CREATE SESSION 
init = tf.global_variables_initializer()

with tf.Session() as sess: 
    sess.run(init)
    
    # We train the model with 1000 steps (epochs)
    for step in range(1000):
        # Tuple unpacking: It returns a tuple with the x_features and labels in bacthes of 100
        # This function is convient, a lot of the times this part is complicated
        batch_x, batch_y = mnist.train.next_batch(batch_size=100)
        
        sess.run(train,feed_dict={x:batch_x, y_true:batch_y})
        
    # EVALUATE THE MODEL
    # This function asks what is the predicted index label with the highest probability
    pred = tf.argmax(y,1)
    # It takes the real label 
    gt = tf.argmax(y_true,1)
    # Check where both tensors are queal
    correct_prediction = tf.equal(pred, gt)
    
    # Transformation of from booleanvariables to numbers (0.0 or 1.0) and calculate the mean over that 
    # Take into account that the larger the True values (1.0s) the better
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    # 'acc' is a graph by itself 
    final_acc = sess.run(acc,feed_dict={x:mnist.test.images, y_true:mnist.test.labels})
    print(f'Final accuracy of the model on test data: {final_acc}')


