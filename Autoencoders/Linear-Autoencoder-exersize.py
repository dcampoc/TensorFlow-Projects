# -*- coding: utf-8 -*-
"""
Apply a linear autoencoder to a 30 feature dataset in order to encode relevant information that guarantees class separability (i.e., feature reduction)

@author: dcamp
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df = pd.read_csv('anonymized_data.csv')
data = np.asanyarray(df)

features = data[:,0:-1]
labels = data[:,-1]

features_scaled = scaler.fit_transform(features)


from tensorflow.keras import models
import tensorflow.keras as keras
from tensorflow.keras import layers

# Sequence of layers
dnn_keras_model = models.Sequential()

# We use linear activation function since it is an autoencoder (no activation function)
dnn_keras_model.add(layers.Dense(units=2, input_dim=30, activation='linear'))

# We use linear activation function since it is an autoencoder (no activation function) 
dnn_keras_model.add(layers.Dense(units=30, activation='linear'))

# Set the layers for the autoencoder
dnn_keras_model.compile(optimizer='adam', loss='MSE', metrics=['MSE'])

# Training
dnn_keras_model.fit(features_scaled, features_scaled,epochs=200)

# Extracting the outputs from the involved layers (one is for the hidden layer and the other one for the output)
extractor = keras.Model(inputs=dnn_keras_model.inputs,
                        outputs=[layer.output for layer in dnn_keras_model.layers])

# Evaluate the layers on the scaled data
features = extractor(features_scaled)

# Open a session for extracting and visualizing the hidden layer outputs (reduced features) 
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()
    hidden_output = features[0].eval()
    plt.scatter(hidden_output[:,0],hidden_output[:,1],c=labels)
    plt.title('Hidden outputs reduced to two features')

print('Note that the reduced features still allow us to differenciate the data')







