# -*- coding: utf-8 -*-
"""
Restore a model in TF (RUN FIRST 'regresions.py')

@author: dcamp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

y_true = (0.5 * x_data) + 5 + noise

# Create graph information
batch_size = 8
m = tf.Variable(np.random.randn(1))
b = tf.Variable(np.random.randn(1))

x_ph = tf.placeholder(tf.float64,[batch_size])
y_ph = tf.placeholder(tf.float64,[batch_size])

y_model = m*x_ph + b
error = tf.reduce_sum(tf.square(y_ph - y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()


saver = tf.train.Saver()
with tf.Session() as sess:
    
    # Restore a model
    saver.restore(sess,'models_saved/simpleLR.ckpt')
    model_m, model_b = sess.run([m,b])

ind_rand = np.random.randint(len(x_data), size=250)
x_plot = x_data[ind_rand]
y_final = x_data[ind_rand]*model_m + model_b

fig_1 = plt.figure()
ax = plt.subplot(111)
ax.plot(x_plot, y_true[ind_rand],'*', label='original data')
ax.plot(x_plot, y_final,'g', label='loaded estiimation')
ax.legend()
plt.ylabel('x', fontsize=16)
plt.xlabel('y', fontsize=16)
plt.title('Rregression using TF')
plt.show()