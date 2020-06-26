# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:55:39 2020
Exercize showing the application of TF.estimator API

@author: dcamp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
tf.disable_v2_behavior()

df = pd.read_csv('cal_housing_clean.csv')
X = df.drop('medianHouseValue', axis=1)
y = df['medianHouseValue']
scaler = MinMaxScaler()
#X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Normalize based only on the training data 
X_train_scaled = scaler.fit_transform(X_train)
# Transform the test (brand new data) based on the normalization done with the training data
X_test_scaled = scaler.transform(X_test) 

# Recreate dataframes from the numpy arrays
df_X_train = pd.DataFrame(data = X_train_scaled, columns=X_train.columns, index=X_train.index)    
    
df_X_test = pd.DataFrame(data = X_test_scaled, columns=X_test.columns, index=X_test.index)    

# Initialize the cpòumn of features    
feat_cols = []
for i in range(df_X_train.shape[1]):
    feat_cols.append(tf.feature_column.numeric_column(df_X_train.columns[i]))

##############################################################################
input_func = tf.estimator.inputs.pandas_input_fn(x=df_X_train,y=y_train,batch_size=8,num_epochs=1000,shuffle=True)

model = tf.estimator.LinearRegressor(feature_columns=feat_cols)
model.train(input_fn=input_func, steps=1000)

test_input_func = tf.estimator.inputs.pandas_input_fn(x=df_X_test, y=y_test, batch_size=10, num_epochs=1,shuffle=False)
results_linear = model.evaluate(test_input_func)

pred_input_func = tf.estimator.inputs.pandas_input_fn(x=df_X_test,batch_size=10,num_epochs=1,shuffle=False)
# Prediction outputs a generator
predictions = model.predict(pred_input_func)
my_pred = list(predictions)
final_pred_LR = []
for pred in my_pred:
    final_pred_LR.append(pred['predictions'])

print("Now let's do a dense NN classifier!".upper())

dnn_model = tf.estimator.DNNRegressor(hidden_units=[6,10,15,20],feature_columns=feat_cols)

dnn_model.train(input_fn=input_func, steps=50000)
results_DNN = dnn_model.evaluate(test_input_func)

# Prediction outputs a generator
predictions_DNN = dnn_model.predict(pred_input_func)
my_pred_DNN = list(predictions_DNN)
final_pred_DNN = []
for pred in my_pred_DNN:
    final_pred_DNN.append(pred['predictions'])


from sklearn.metrics import mean_squared_error
MSE_LR = mean_squared_error(y_true= y_test,y_pred=final_pred_LR)
MSE_DNN = mean_squared_error(y_true= y_test,y_pred=final_pred_DNN)

print('Results of the linear regressor'.upper())
print(results_linear)
print('\n')
print('Results of the DNN'.upper())
print(results_DNN)
print('\n')

if results_linear['average_loss'] < results_DNN['average_loss']:
    print('The linear classifier performed better than the DNN')
else:
    print('The DNN performed better than the linear classifier')

print(f"DNN's loss: {results_DNN['average_loss']}")
print(f"Linear regressor's loss: {results_linear['average_loss']}")
print('\n')
print(f"Linear regressor's MSE: {MSE_LR}")
print(f"DNN's MSE: {MSE_DNN}")



    
    

