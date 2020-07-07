# -*- coding: utf-8 -*-
"""
Example of classification by using the the estimator API

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

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import estimator


print(f'Train shape: {X_train.shape}')

# Create (in a fast way without for loops) a list in tf of numeric columns coming from the training features
feat_cols = [tf.feature_column.numeric_column('x',shape=[13])]

# Create a neural network for classification purposes 
deep_model = estimator.DNNClassifier(hidden_units=[13, 13, 13],\
                                     feature_columns=feat_cols, n_classes=3, optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01))

input_fn = estimator.inputs.numpy_input_fn(x={'x':scaled_x_train}, y=y_train, shuffle=True, batch_size=10, num_epochs=5) 

deep_model.train(input_fn, steps=500)

input_fn_eval = estimator.inputs.numpy_input_fn(x={'x':scaled_x_test}, shuffle=False) 

# Cast predictions in a list
preds = list(deep_model.predict(input_fn=input_fn_eval))
# deep_model.evaluate() ---> it uses the testing labels

predictions = [p['class_ids'][0] for p in preds]

from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test, predictions))
print('\n')
print('Confusion matrix'.upper())
print(confusion_matrix(y_test, predictions))
