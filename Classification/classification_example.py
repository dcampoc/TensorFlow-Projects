# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 19:17:21 2020
Classfication example using TF API (estimator). A linear classifier and a densely neural network (fully-connected NN) are compared

@author: dcamp
"""
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

diabetes = pd.read_csv('pima-indians-diabetes.csv')
diabetes.head(5)
diabetes.columns
# Class (0): The person has diabetes
# Class (1): The person does not have diabetes
# Group is a created (artificial) categorical feature

cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']

#Normalizing in one line with pandas
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# In case we know the categorical values of the column
assigned_group_1 = tf.feature_column.categorical_column_with_vocabulary_list('Group', \
            ['A', 'B', 'C', 'D'])
# In case we don't know the categorical values of the column or there are too many of them, has_bu_buscket_size should be higher than the total of expected categorical values
assigned_group_2 = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=30)

# Feature engieneering: convert features from one space to another in  order to extract more information out of them
diabetes['Age'].hist(bins=25)

age_bucket = tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80])
feat_cols = [num_preg ,plasma_gluc,dias_press,tricep ,insulin,bmi,diabetes_pedigree,assigned_group_2, age_bucket]

# Separating the featires from the class
x_data = diabetes.drop('Class', axis=1)
x_data.head()
labels = diabetes['Class']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3)

input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
model.train(input_fn=input_func, steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=10, num_epochs=1,shuffle=False)
results_linear = model.evaluate(eval_input_func)

pred_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test,batch_size=10,num_epochs=1,shuffle=False)
predictions = model.predict(pred_input_func)
my_pred = list(predictions)

print('See for example the results for the first testing sample:')
print(my_pred[0])
print('\n')
print(f"The probability of the first sample to belong to each class is: {my_pred[0]['probabilities']}")
print('\n')

print("Now let's do a dense NN classifier!".upper())

#dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols, n_classes=2)

# The densely NN cannot work with the categorical column without first being embedded
embedded_group_column = tf.feature_column.embedding_column(assigned_group_2, dimension=4)
# Reset the feature columns based on the embedded new column 
feat_cols = [num_preg ,plasma_gluc,dias_press,tricep ,insulin,bmi,diabetes_pedigree,embedded_group_column, age_bucket]

dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols, n_classes=2)

dnn_model.train(input_fn=input_func, steps=1000)
results_DNN = dnn_model.evaluate(eval_input_func)
predictions_DNN = dnn_model.predict(pred_input_func)
my_pred_DNN = list(predictions_DNN)

print('See for example the results for the first testing sample:')
print(my_pred_DNN[0])
print('\n')
print(f"The probability of the first sample to belong to each class is: {my_pred_DNN[0]['probabilities']}")
print('\n')
print('\n')
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
