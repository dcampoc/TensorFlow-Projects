# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:05:15 2020

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
x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])
x_df.head()
y_df.head()
# concatenate data frames 
my_data = pd.concat([x_df,y_df], axis=1)

# Returning 250 random samples from the million 
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y', title='Initial data')

print('Million points cannot be trained at once!'.upper())
print('\n')
print('It is necessary to use batches of data'.upper())

# There is no true or false for selecting a batch size
batch_size = 8
# This are the variables to adjust
m = tf.Variable(np.random.randn(1))
b = tf.Variable(np.random.randn(1))

# x_ph is my place holder (variable 'x')
x_ph = tf.placeholder(tf.float64,[batch_size])
y_ph = tf.placeholder(tf.float64,[batch_size])

y_model = m*x_ph + b
error = tf.reduce_sum(tf.square(y_ph - y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    m_init = m.eval()
    b_init = b.eval()
    # Number of batches of dimension 8    
    bacthes = 1000
    
    for i in range(bacthes):
        # Generate 8 random integers at each iteration
        rand_ind = np.random.randint(len(x_data),size=batch_size)
        # It indicates input and outputs (supervised learning)
        feed = {x_ph:x_data[rand_ind],y_ph:y_true[rand_ind]}
        sess.run(train,feed_dict=feed)
    
    model_m, model_b = sess.run([m,b])
    
 
ind_rand = np.random.randint(len(x_data), size=250)
x_plot = x_data[ind_rand]
y_init = x_data[ind_rand]*m_init + b_init
y_final = x_data[ind_rand]*model_m + model_b

fig_1 = plt.figure()
ax = plt.subplot(111)
ax.plot(x_plot, y_true[ind_rand],'*', label='original data')
ax.plot(x_plot, y_init,'r', label='initial estiimation')
ax.plot(x_plot, y_final,'g', label='final estiimation')
ax.legend()
plt.ylabel('x', fontsize=16)
plt.xlabel('y', fontsize=16)
plt.title('Rregression using TF')
plt.show()


print('\n')
print('Tf estimator'.upper())

# Note that the shape has only one dimension
feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

from sklearn.model_selection import train_test_split
# A random state is set for reproducibility purposes
x_train, x_test, y_train, y_test = train_test_split(x_data, y_true, test_size=0.3,\
                                                    random_state= 101 )
print(f'The shae of the training data is: {x_train.shape}')
print(f'The shape of my testing data is: {x_test.shape}')

# Note that the estimator inputs can be in np or pd formats 
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train,\
                                                batch_size=8, num_epochs=None, shuffle=True)
#input_func = tf.estimator.inputs.pandas_input_fn(...)

train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train,\
                                                batch_size=8, num_epochs=1000, shuffle=False)

test_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_test}, y_test,\
                                                batch_size=8, num_epochs=1000, shuffle=False)

# We did not specify the number of epochs or steps, we will put it here
estimator.train(input_fn=input_func,steps=1000)
# As this is for evaluation there is no shuffling to see hw it really works
train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)
# Our test metrics come as follows
test_metrics = estimator.evaluate(input_fn=test_input_func, steps=1000)

# Let's print the metrics for the training and testing
print('training data metrics'.upper())
print(train_metrics)

print('test data metrics'.upper())
print(test_metrics)
# Remember that one way to detect overfitting is when you have a low loss in the training data but a large loss in the testing data 

brand_new_data = np.linspace(0,10,10)
input_fn_predict =  tf.estimator.inputs.numpy_input_fn({'x':brand_new_data}, shuffle=False)

brand_new_data_estimation = list(estimator.predict(input_fn=input_fn_predict))
# The predict method outputs an iterator at each position it carries a key: 'predictions' and the value. See the following 2 commented lines 
# brand_new_data_estimation = estimator.predict(input_fn=input_fn_predict)
# next(bran_new_data_estimation)

predictions= []
for pred in estimator.predict(input_fn=input_fn_predict):
    
    predictions.append(pred['predictions'])
    

fig_2 = plt.figure()
ax = plt.subplot(111)
ax.plot(x_plot, y_true[ind_rand],'*', label='original data')
ax.plot(brand_new_data,predictions, 'r*', label='predicted points')
ax.legend()
plt.ylabel('x', fontsize=16)
plt.xlabel('y', fontsize=16)
plt.title('Estimation using TF')
plt.show()

