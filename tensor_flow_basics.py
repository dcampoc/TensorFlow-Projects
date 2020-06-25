# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 01:20:07 2020

@author: dcamp
"""

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
print(f'tensorflow version: {tf.__version__}')
sess = tf.InteractiveSession()
hello = tf.constant('hello')
world = tf.constant(' world')
type(hello)
print(hello)
#tf.print(hello + world)
print(sess.run(hello + world))

a = tf.constant(10)
b = tf.constant(5)   
#tf.print(a*b)
#tf.print(a+b) 
#tf.print(a-b)
print(sess.run(a * b))
print(sess.run(a + b))
print(sess.run(a - b))

# fill a matrix full of 10s
fill_mat = tf.fill((4,4),10)
#tf.print(fill_mat)
print(sess.run(fill_mat))
print('\n')
# fill a matrix full of 0s
my_zeros = tf.zeros((4,4))
#tf.print(my_zeros)
print(sess.run(my_zeros))
print('\n')
# fill a matrix full of 1s
my_ones = tf.ones((4,4))
#tf.print(my_ones)
print(sess.run(my_ones))
print('\n')
# fill a matrix with elements coming from a random distribution mean = 0, std = 2
my_randn = tf.random.normal((4,4),mean=10,stddev=2)
#tf.print(my_randn)
print(sess.run(my_randn))
print('\n')
# fill a matrix with elements coming from a uniform distribution between values 3 and 5
my_randu = tf.random.uniform((4,4), minval=3, maxval=5)
#tf.print(my_randu)
print(sess.run(my_randu))
print('\n')
# multiply two matrices element-wise
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[3,6],[3,5]])
c = a*b
#tf.print(c)
print(sess.run(c))
print(f"The shape of the final matrix is {c.get_shape()}")
print('\n')
# Multiply two matrices normally
b_2 = tf.constant([[3],[5]])
c_2 = tf.matmul(a,b_2)
#tf.print(c_2)
print(sess.run(c_2))
tf.InteractiveSession.close(sess)

###############################################################
# Graphs and swithing among them
n_1 = tf.constant(3)
n_2 = tf.constant(2)
n_3 = n_1 + n_2

with tf.Session() as sess:
    result = sess.run(n_3)
print(result)
# print deafult grph (related to the default session we are working on right now)
print(tf.get_default_graph())
# Create a new graph
g = tf.Graph()
# Print its position in memory
print(g) 
# Activating locally 'g' as default graph
with g.as_default():
    print(tf.get_default_graph())
    
my_tensor = tf.random.uniform((4,4),0,1)
ph = tf.placeholder(tf.float32, shape=(4,4))
# A lot of times placeholders have none because it carry the actual number samples in data (we may not know this since they usually come in batches) 
ph_1 = tf.placeholder(tf.float32, shape=(None,4))
my_var = tf.Variable(my_tensor)
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(my_var))

##########################################################################
# Simple neural network stricture (no training) taking 10 features
n_features = 10
n_dense_neurons = 3 

# Create a placeholder for the input ('None' refers to the fact that there is no knowledge about the number of samples that we are going to get)
x = tf.placeholder(tf.float32,(None,n_features))
# Weights are created
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
# after 'x' multiplies 'W', the results gets summed by 'b' which is related to the bias term 
b = tf.Variable(tf.ones([n_dense_neurons]))

xW = tf.matmul(x,W)
z = tf.add(xW,b)
a = tf.sigmoid(z)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    layer_out = sess.run(a,feed_dict={x:np.random.random([1,n_features])})
    print(layer_out)

# Simple linear regression example
print('Simple regression example'.upper())
# Create linear trend
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5,10) 
import matplotlib.pyplot as plt 
plt.plot(x_data, y_label, '*')
plt.show()

# create y = mx + b

m = tf.Variable(np.random.rand(1))
#m = 0.40098911
b = tf.Variable(np.random.rand(1))
#b = 0.98741185
init_m = m
init_b = b

error = 0

for x,y in zip(x_data, y_label):
    y_hat = m*x + b
    
    error += (y - y_hat)**2
    
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    training_steps = 100
    for i in range(training_steps):
        sess.run(train)
    final_slope, final_intercept = sess.run([m,b])
    
x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept
y_pred_init = init_m*x_test + init_b

plt.plot(x_data, y_label, '*')
with tf.Session() as sess:
    sess.run(init)
    plt.plot(x_test, y_pred_init.eval(), 'r')     
plt.plot(x_test, y_pred_plot,'green')

plt.show()
    


