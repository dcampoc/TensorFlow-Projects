# -*- coding: utf-8 -*-
"""
Exercise that compares linear classification and a DNN classifier on dataset involving numerical and categorical data 

@author: dcamp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
import copy
tf.disable_v2_behavior()

df = pd.read_csv('census_data.csv')
df['income_bracket'] = df['income_bracket'].str.replace(' >50K','1')
df['income_bracket'] = df['income_bracket'].str.replace(' <=50K','0')
df['income_bracket'] = df['income_bracket'].astype(np.int64)

X = df.drop('income_bracket', axis=1)
y = df['income_bracket']
#scaler = MinMaxScaler()
#X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Divide the features into categorical and numerical
cat_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country']
num_features = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

'''
# Normalize based only on the training data 
X_train_scaled = scaler.fit_transform(X_train)
# Transform the test (brand new data) based on the normalization done with the training data
X_test_scaled = scaler.transform(X_test) 

# Recreate dataframes from the numpy arrays
df_X_train = pd.DataFrame(data = X_train_scaled, columns=X_train.columns, index=X_train.index)    
    
df_X_test = pd.DataFrame(data = X_test_scaled, columns=X_test.columns, index=X_test.index)    
'''

# In case we don't know the categorical values of a column or there are too many of them. A hash bucket is used. hash_bucket_size should be higher than the expected number of categorical values 
assigned_groups = []
categories_dimension = []
for feat in cat_features:  
    assigned_groups.append(tf.feature_column.categorical_column_with_hash_bucket(feat, hash_bucket_size=32561))
    categories_dimension.append(len(df[feat].unique()))

# Initialize the column of features    
feat_cols = copy.deepcopy(assigned_groups)
for feat in num_features:
    feat_cols.append(tf.feature_column.numeric_column(feat))

##############################################################################
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=8,num_epochs=1000,shuffle=True)

model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
model.train(input_fn=input_func, steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1,shuffle=False)
results_linear = model.evaluate(eval_input_func)

pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)
predictions = model.predict(pred_input_func)
my_pred = list(predictions)

'''
final_pred_LR = []
for pred in my_pred:
    final_pred_LR.append(pred['predictions'])
'''

print("Now let's do a dense NN classifier!".upper())
# The densely NN cannot work with the categorical column without first being embedded
embedded_group_columns = []
i = 0
for assigned_group in assigned_groups:
    embedded_group_columns.append(tf.feature_column.embedding_column(assigned_group, dimension=categories_dimension[i]))  
    i += 1

feat_cols = copy.deepcopy(embedde
                          d_group_columns)
for feat in num_features:
    feat_cols.append(tf.feature_column.numeric_column(feat)) 

dnn_model = tf.estimator.DNNClassifier(hidden_units=[14,20,20,20],feature_columns=feat_cols,n_classes=2)
dnn_model.train(input_fn=input_func, steps=1000)
results_DNN = dnn_model.evaluate(eval_input_func)
predictions_DNN = dnn_model.predict(pred_input_func)
my_pred_DNN = list(predictions_DNN)

print('Results of the linear classifier'.upper())
print(results_linear)
print('\n')
print('Results of the DNN'.upper())
print(results_DNN)
print('\n')

if results_linear['accuracy'] >= results_DNN['accuracy']:
    print('The linear classifier performed better than the DNN')
else:
    print('The DNN performed better than the linear classifier')

print(f"DNN's accuracy: {results_DNN['accuracy']}")
print(f"Linear classifier's accuracy: {results_linear['accuracy']}")

print('\n')
print('Classification reports'.upper())
print('\n')
from sklearn.metrics import classification_report
# Using list comprehension for unwrapping predictions of classifiers
y_pred_linear = [pred['class_ids'][0] for pred in my_pred]
y_pred_DNN = [pred['class_ids'][0] for pred in my_pred_DNN]
print('Linear classifier'.upper())
print(classification_report(y_true=y_test, y_pred=y_pred_linear))
print('\n')
print('DNN classifier'.upper())
print(classification_report(y_true=y_test, y_pred=y_pred_DNN))


