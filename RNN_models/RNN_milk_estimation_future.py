# -*- coding: utf-8 -*-
"""
Prediction of milk production based on real data using RNNs 
Estimations are done by RNNs that predict following year milk's production

Already trained LSTM and GRU networks can be found in the folder 'API_trained_models' 
Networks' names: 'rnn_milk_approx_GRU_future' and 'rnn_milk_approx_LSTM_future' 

@author: dcamp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.preprocessing import MinMaxScaler

milk = pd.read_csv('monthly-milk-production.csv',index_col='Month')

milk.head()

milk.index = pd.to_datetime(milk.index)

# Pandas automatically interprets the x-axis as time when plotting it (see the data)
milk.plot()

milk.info()

# We will train with the whole data except the year. 
# The last 12 months of data will be used for testing purposes 
train_set = milk.head(156)

test_set = milk.tail(12)


# SCALE THE DATA
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)

# CREATION OF BATCH FUNCTIONS
def next_batch(training_data,steps,steps_ahead):  
    
    # Grab a random starting point for each batch
    rand_start = np.random.randint(0,len(training_data)-(steps*steps_ahead))

    # Create inputs and output batches
    
    x_batch = np.array(training_data[rand_start:rand_start+steps]).reshape(-1, steps, 1)     
    
    y_batch = np.array(training_data[rand_start+steps:rand_start+(steps*steps_ahead)]).reshape(-1, steps, 1) 
    
    return x_batch, y_batch 

# PARAMETERS OF THE RNN
# Just one feature, the time series
num_inputs = 1
# Num of steps in each batch (12 months)
num_time_steps = 12
# Number of neurons
num_neurons = 500
# Just one output, predicted time series
num_outputs = 1

## You can also try increasing iterations, but decreasing learning rate
# learning rate you can play with this
learning_rate = 0.001 
# how many iterations to go through (training steps), you can play with this
num_train_iterations = 10000
# Size of the batch of data
batch_size = 1

# CREATION OF THE RNN
x = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

#cell = tf.nn.rnn_cell.GRUCell(num_units=num_neurons, activation=tf.nn.tanh)
cell = tf.nn.rnn_cell.LSTMCell(num_units=num_neurons, activation=tf.nn.tanh)

outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

# MSE
loss = tf.reduce_mean(tf.square(outputs-y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# SESSION 

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

saver = tf.train.Saver()

with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as sess:
    
    sess.run(init)
    
    for iteration in range(num_train_iterations):
        
        x_batch, y_batch = next_batch(train_scaled, num_time_steps, 2)
        
        sess.run(train, feed_dict={x:x_batch, y:y_batch})
        
        if iteration %100 == 0:
            
            mse = loss.eval(feed_dict={x:x_batch, y:y_batch})
            print(iteration,'\tMSE',mse)

    #saver.save(sess, "./rnn_milk_approx_GRU_future")                # COMMENT THIS LINE FOR SAVING LSTM MODEL
    saver.save(sess, "./rnn_milk_approx_LSTM_future")  
    
print('This is the test set:'.upper())
print(test_set)




with tf.Session() as sess:
    
    # Use your Saver instance to restore your saved rnn time series model
    #saver.restore(sess, "./rnn_milk_approx_LSTM_future")
    saver.restore(sess, "./rnn_milk_approx_LSTM_future") 
    
    # Put the entire training set except for the last 12 months
    X_batch = train_scaled[-12:].reshape(-1, num_time_steps, num_inputs)
    y_pred = sess.run(outputs, feed_dict={x: X_batch})
    final_pred = np.append(train_scaled, y_pred[0, :, 0])

# The results should be "unnormalized" 
final_pred = scaler.inverse_transform(final_pred.reshape(-1,1))
ground_truth = np.array(milk['Milk Production'])

plt.figure()
plt.plot(ground_truth, 'g', label='Ground truth')
plt.plot(final_pred, 'k', label='Predictions')
plt.legend()
plt.show()



