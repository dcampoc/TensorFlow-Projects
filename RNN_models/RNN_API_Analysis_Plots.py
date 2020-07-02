# -*- coding: utf-8 -*-
"""
Load and test of RNNs (GRU or LSTM) produced with the algorithm "RNN_API.py". 
Please run "RNN_API.py" first or copy and paste the trained models (located at the folder "API_trained_models") in the current folder

@author: dcamp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# We only have a single feaure in this time series from x to y = sin(x) 
num_inputs = 1
num_outputs = 1

# Let's suppose each random batches has 30 steps in it
num_time_steps = 30

minX = 0
maxX = 50
num_neurons = 100 # 200 for LSTM (it takes longer for producing results)
# x-data values to evaluate the RNNs
x_tot = np.arange(minX, maxX, 0.08)

# delete the current graph
tf.reset_default_graph()

# PLACEHOLDERS

x = tf.placeholder(tf.float32,[None,num_time_steps, num_inputs])
y = tf.placeholder(tf.float32,[None,num_time_steps, num_outputs])


cell = tf.nn.rnn_cell.GRUCell(num_units=num_neurons, activation=tf.nn.tanh)
#cell = tf.nn.rnn_cell.LSTMCell(num_units=num_neurons, activation=tf.nn.tanh) 

outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

# MSE (Note that this part is related to the training and IT IS NOT NECESSARY)
#loss = tf.reduce_mean(tf.square(outputs-y))

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_Rate)

#train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    
    # SELECT WHAT NETWORK IS GOING TO BE RESTORED 
    saver.restore(sess,"./rnn_sin_approx_GRU")           # COMMENT THIS LINE FOR LSTM
    #saver.restore(sess,"./rnn_sin_approx_LSTM_5K_iter") # UNCOMMENT THIS LINE FOR LSTM
    y_pred_tot = np.sin(x_tot[0:30])
    y_pred_tot_2 = []
    
    for iteration in range(len(x_tot) - num_time_steps -1):
        x_chunk = x_tot[iteration:iteration+num_time_steps]
        
        x_batch = np.sin(x_chunk).reshape(-1, num_time_steps,num_inputs)
        y_pred = sess.run(outputs, feed_dict={x:x_batch})
        y_pred_tot = np.append(y_pred_tot, y_pred[0,-1,0])
        
        
        x_batch_2 = (np.sin(x_chunk)/(x_chunk+0.1)).reshape(-1, num_time_steps,num_inputs)
        y_pred2 = sess.run(outputs, feed_dict={x:x_batch_2})
        y_pred_tot_2 = np.append(y_pred_tot_2, y_pred2[0,-1,0])
        
plt.figure()
plt.plot(x_tot[:-1], np.sin(x_tot[:-1]), 'go', markersize=6, alpha=0.5, label='Original data')
plt.plot(x_tot[1:], y_pred_tot, 'ko', markersize=4, alpha=0.5, label='Predictions (next time instant)')
plt.legend()
plt.title('sin(x) as reference and RNN (GRU or LSTM) predicting nect time instances')
plt.tight_layout()


plt.figure()
plt.plot(x_tot[30:-1], np.sin(x_tot[30:-1])/x_tot[30:-1], 'go', markersize=6, alpha=0.5, label='Original data')
plt.plot(x_tot[31:], y_pred_tot_2, 'ko', markersize=4, alpha=0.5, label='Predictions (next time instant)')
plt.legend()
plt.title('sinc(x) as reference and RNN (GRU or LSTM) predicting nect time instances')
plt.tight_layout()

print('Note that in both plot cases the RNN is trained based on the sin(x) function')


