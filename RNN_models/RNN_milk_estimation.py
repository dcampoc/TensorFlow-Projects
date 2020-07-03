# -*- coding: utf-8 -*-
"""
Prediction of milk production based on real data using RNNs 
Estimations are done by RNNs that predict only the following time instant (a single moth) 
We aim at predicting a whole year. For a more accurate RNN structure see 'RNN_milk_estimation_future'

Already trained LSTM and GRU networks can be found in the folder 'API_trained_models' 
Networks' names: 'rnn_milk_approx_GRU' and 'rnn_milk_approx_LSTM' 

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
def next_batch(training_data,batch_size,steps):  
    
    # Grab a random starting point for each batch
    rand_start = np.random.randint(0,len(training_data)-steps) 

    # Create Y data for time series in the batches
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)

    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1) 

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

cell = tf.nn.rnn_cell.GRUCell(num_units=num_neurons, activation=tf.nn.tanh)    # COMMENT THIS LINE FOR USING LSTM MODEL
# cell = tf.nn.rnn_cell.LSTMCell(num_units=num_neurons, activation=tf.nn.tanh) # UNCOMMENT THIS LINE FOR USING LSTM MODEL

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
        
        x_batch, y_batch = next_batch(train_scaled, batch_size, num_time_steps)
        
        sess.run(train, feed_dict={x:x_batch, y:y_batch})
        
        if iteration %100 == 0:
            
            mse = loss.eval(feed_dict={x:x_batch, y:y_batch})
            print(iteration,'\tMSE',mse)

    saver.save(sess, "./rnn_milk_approx_GRU")                                   # COMMENT THIS LINE FOR USING LSTM MODEL
    # saver.save(sess, "./rnn_milk_approx_LSTM")                                # UNCOMMENT THIS LINE FOR USING LSTM MODEL
    
print('This is the test set:'.upper())
print(test_set)


with tf.Session() as sess:
    
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "./rnn_milk_approx_GRU")                                # COMMENT THIS LINE FOR USING LSTM MODEL
    # saver.restore(sess, "./rnn_milk_approx_LSTM")                             # UNCOMMENT THIS LINE FOR USING LSTM MODEL

    # Put the entire training set except for the last 12 months
    final_pred = train_scaled
    X_batch = train_scaled[-12:].reshape(-1, num_time_steps, num_inputs)
    pred_cum = np.array([])
    # Now create a for loop that generates predictions (estimating one month in the future) by taking last predictions for 12 months  
    for i in range(12):  
        y_pred = sess.run(outputs, feed_dict={x: X_batch})
        final_pred = np.append(final_pred, y_pred[0, -1, 0])
        pred_cum = np.append(pred_cum, y_pred[0, -1, 0])
        X_batch = np.append(train_scaled[(-12 + (i+1)):], pred_cum)
        X_batch = X_batch.reshape(-1, num_time_steps, num_inputs)


# The results should be "unnormalized" 
final_pred = scaler.inverse_transform(final_pred.reshape(-1,1))
ground_truth = np.array(milk['Milk Production'])

plt.figure()
plt.plot(ground_truth, 'g', label='Ground truth')
plt.plot(final_pred, 'k', label='Predictions')
plt.legend()
plt.show()

print('Note that the model only generates a single month prediction, it is unfair to ask for 12 months sequentially'.upper())



