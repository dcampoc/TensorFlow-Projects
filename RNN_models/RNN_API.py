# -*- coding: utf-8 -*-
"""
RNN training and partial evaluation by using the tensorflow API

@author: dcamp
"""

# We will provide a time series as input
# As output, we will predict the time series shifted one time instant in the future 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Class to create data and generate batches of data
class TimeSeriesData():
    
    def __init__(self,num_points,xmin,xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax - xmin)/num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        # This is the ground truth (We will predict a sine function!)
        self.y_true = np.sin(self.x_data)
        
    # Convinient method to transform any x data to sin(x)
    def ret_true(self,x_series):
        return np.sin(x_series)
    
    def next_batch(self,batch_size,steps, return_batch_ts=False):
        
        # Grab a random starting point for each batch
        rand_start = np.random.rand(batch_size,1)
        
        # Convert to be a time series  (we need to convert the random start point to be in random series)
        ts_start = self.xmin + rand_start * (self.xmax - self.xmin - (steps*self.resolution))
        #ts_start = rand_start * (self.xmax - self.xmin - (steps*self.resolution))

        # Create batch time series an the x-axis
        batch_ts = ts_start + np.arange(0.0, steps+1)* self.resolution
        
        # Create the Y data for the time series x-axis from previous step       
        y_batch = np.sin(batch_ts)       
        
        #Formatting for RNN
        if return_batch_ts:
            return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1), batch_ts
        else:
            # The first batch represents the time series and the second represents the time series shited over by one step to the future 
            return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1)
        

ts_data = TimeSeriesData(250,0,20)
plt.figure()
plt.plot(ts_data.x_data, ts_data.y_true,'r*')
plt.title('Initial GT data')

# Let's suppose each of my random batches to have 30 steps in it
num_time_steps = 30

y1, y2, ts = ts_data.next_batch(batch_size=1, steps=num_time_steps, return_batch_ts=True)
print(f"Shape of temporal batch {ts.shape}, it must be flatten for plotting")

plt.figure()
plt.plot(ts_data.x_data, ts_data.y_true,'r', label='sin(t)')
plt.plot(ts.flatten()[:-1], y1.flatten(),'b*', label='single input')
plt.title('GT data and input data')
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(ts_data.x_data, ts_data.y_true,'r', label='sin(t)')
plt.plot(ts.flatten()[1:], y2.flatten(),'b*', label='single output')
plt.title('GT data and output data (shifted forward in time)')
plt.legend()
plt.tight_layout()

# TRAINING DATA
train_inst = np.linspace(5,5+ts_data.resolution*(num_time_steps+1), num_time_steps+1)

plt.figure()
plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), 'go', markersize=15, alpha=0.5, label='training input instance')

plt.plot(train_inst[1:], ts_data.ret_true(train_inst[1:]), 'ko', markersize=5, alpha=1, label='training output instance')

plt.legend()
plt.tight_layout()

print('Given the green points, can it predict the black dots?'.upper())
print('The RNN will try to do that'.upper())

tf.reset_default_graph()
# We only have a single feaure in this time series from x to y = sin(x) 
num_inputs = 1
num_neurons = 100 # 200 for LSTM
num_outputs = 1
learning_Rate = 0.01 #0.001 for LSTM
num_train_iterations = 1000 # 5000 for LSTM
batch_size = 1

# PLACEHOLDERS

x = tf.placeholder(tf.float32,[None,num_time_steps, num_inputs])
y = tf.placeholder(tf.float32,[None,num_time_steps, num_outputs])

# CREATE RNN CELL LAYER (options: Basic RNN and LSTM cells, multi RNN and LSTM cells)
# We will deal with a basic RNN cell 

cell = tf.nn.rnn_cell.GRUCell(num_units=num_neurons, activation=tf.nn.tanh)
#cell = tf.nn.rnn_cell.LSTMCell(num_units=num_neurons, activation=tf.nn.tanh) 

outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

# MSE
loss = tf.reduce_mean(tf.square(outputs-y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_Rate)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()


# SESSION 

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

saver = tf.train.Saver()

with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as sess:
    
    sess.run(init)
    
    for iteration in range(num_train_iterations):
        
        x_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
        
        sess.run(train, feed_dict={x:x_batch, y:y_batch})
        
        if iteration %100 == 0:
            
            mse = loss.eval(feed_dict={x:x_batch, y:y_batch})
            print(iteration,'\tMSE',mse)
    
    #   SAVE IT IN LSTM WHEN WORKING WITH IT (IT NEEDS AROUND 5K ITERATIONS, LR: 0.001 AND 200 NEURONS)
    saver.save(sess, "./rnn_sin_approx_GRU")                # COMMENT THIS LINE FOR SAVING LSTM MODEL
    # saver.save(sess, "./rnn_sin_approx_LSTM_5K_iter")     # UNCOMMENT THIS LINE FOR SAVING LSTM MODEL
    
    # LOCAL PARTIAL TEST. CHECK RNN_test_plot.py FOR  
with tf.Session() as sess:
    
    saver.restore(sess,"./rnn_sin_approx_GRU")              # COMMENT THIS LINE FOR CALLING LSTM MODEL
    # saver.restore(sess, "./rnn_sin_approx_LSTM_5K_iter")  # UNCOMMENT THIS LINE FOR CALLING LSTM MODEL
    
    x_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps,num_inputs)))
    y_pred = sess.run(outputs, feed_dict={x:x_new})
    
    train_inst = np.linspace(0,20,500)
    x_new_2 = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps,num_inputs)))
    y_pred_2 = sess.run(outputs, feed_dict={x:x_new})
        

plt.figure()
plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), 'go', markersize=15, alpha=0.5, label='training input instance')

plt.plot(train_inst[1:], ts_data.ret_true(train_inst[1:]), 'ko', markersize=5, alpha=1, label='targets')

plt.plot(train_inst[1:], y_pred[0,:,0], 'bo', markersize=5, alpha=1, label='predictions')

    