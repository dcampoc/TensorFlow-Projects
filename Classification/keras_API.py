# -*- coding: utf-8 -*-
"""
Densily connected neural network using keras API

@author: dcamp
"""

from sklearn.datasets import  load_wine

wine_data = load_wine()

type(wine_data)

print('it behaves as a dictionary'.upper())

print(wine_data.keys())

print(wine_data['DESCR'])

# Taking the feature data and labels

feat_data = wine_data['data']

labels = wine_data['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(feat_data, labels, test_size=0.33, random_state=42)
    
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_x_train = scaler.fit_transform(X_train)

scaled_x_test = scaler.fit_transform(X_test)

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#from tensorflow.compat.v1 import estimator

from tensorflow.keras import models

# Sequence of layers
dnn_keras_model = models.Sequential()

from tensorflow.keras import layers
dnn_keras_model.add(layers.Dense(units=13, input_dim=13, activation='relu'))

# Because they are not the input layers, we do not need to define input_dim 
dnn_keras_model.add(layers.Dense(units=13, activation='relu'))
dnn_keras_model.add(layers.Dense(units=13, activation='relu'))

# Last layer one-hot encoding classes (3 classes)
dnn_keras_model.add(layers.Dense(units=3, activation='softmax'))

# Note: To see the list of losses, optimizers, metrics and activation functions then see the autocompletion of the imported classes. E.g., try optimizers.---> Then see the autocompletion
from tensorflow.keras import losses, optimizers, metrics, activations

# This sets all the layers as they should be
dnn_keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

dnn_keras_model.fit(scaled_x_train,y_train,epochs=50)
predictions = dnn_keras_model.predict_classes(scaled_x_test)

from sklearn.metrics import classification_report
print(classification_report(predictions,y_test))
print('0, 1 and 2 in the table below refer to the 3 considered classes')


