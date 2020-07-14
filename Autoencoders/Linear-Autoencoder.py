# -*- coding: utf-8 -*-
"""
Linear Variational Autoencoder (single hidden layer autoencoder). Equivalent to PCA

The idea is to project a dataset of three dimensions into two (dimensionality reduction) and show how it is possible to classify data into two different groups
@author: dcamp
"""

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Import library for creating classification datasets 
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=100,n_features=3,centers=2, random_state=101)
print(f'The dataset is of the type: {type(data)}')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[0])

# Extraction of three scaled features (columns)
data_x = scaled_data[:,0]

data_y = scaled_data[:,1]

data_z = scaled_data[:,2]

# For creating 3 dimensional plots (matplotlib is not the best library for 3D visualization but it is possible to do it)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
# Color data based on the two classes in data[1]
ax.scatter(data_x, data_y, data_z, c=data[1])




from tensorflow.keras import models
import tensorflow.keras as keras
# Sequence of layers
dnn_keras_model = models.Sequential()

from tensorflow.keras import layers
# We use linear activation function since it is an autoencoder (no activation function)
dnn_keras_model.add(layers.Dense(units=2, input_dim=3, activation='linear'))

# We use linear activation function since it is an autoencoder (no activation function) 
dnn_keras_model.add(layers.Dense(units=3, activation='linear'))

# Set the layers for the autoencoder
dnn_keras_model.compile(optimizer='adam', loss='MSE', metrics=['MSE'])

# Training
dnn_keras_model.fit(scaled_data,scaled_data,epochs=200)

# Extracting the outputs from the involved layers (one is for the hidden layer and the other one for the output)
extractor = keras.Model(inputs=dnn_keras_model.inputs,
                        outputs=[layer.output for layer in dnn_keras_model.layers])

# Evaluate the layers on the scaled data
features = extractor(scaled_data)

# Open a session for extracting and visualizing the hidden layer outputs (reduced features) 
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()
    hidden_output = features[0].eval()
    plt.scatter(hidden_output[:,0],hidden_output[:,1],c=data[1])
    plt.title('Hidden outputs reduced to two features')

print('Note that the reduced  features still allow us to differenciate the data')

predictions = dnn_keras_model.predict(scaled_data)
fig = plt.figure()
plt.title('Estimation of three features by the autoencoder')
ax = fig.add_subplot(111,projection='3d')
# Color data based on the two classes in data[1]
ax.scatter(predictions[:,0], predictions[:,1], predictions[:,2], c=data[1])





