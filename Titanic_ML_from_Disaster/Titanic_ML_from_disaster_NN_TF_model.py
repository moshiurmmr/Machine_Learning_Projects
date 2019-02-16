#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 09:40:02 2019

@author: moshiur
"""

import pandas as pd
import numpy as np

# import custom functions
from removeUniqueFeature import removeUniqueFeature
from imputeWithAvgValue import imputeWithAvgValue
from oneHotEncode import ohe_categorical_data

# import tensorflow
import tensorflow as tf

# import accuracy_score metric
from sklearn.metrics import accuracy_score

# import train-test split model
from sklearn.model_selection import train_test_split

# import data
trainData = pd.read_csv("titanic_data/train.csv")
testData = pd.read_csv("titanic_data/test.csv")

#####   data preparation ##########
# features with null value
trainData.apply(lambda x: x.isnull().any())

# select features
selectFeatures = list(trainData.columns.values)
targetCol = 'Survived'
selectFeatures.remove(targetCol)

# remove features with unique values
selectFeatures = removeUniqueFeature(trainData, selectFeatures)

# remove features that are not significant
selectFeatures.remove('Cabin')
selectFeatures.remove('Ticket')

#
trainData_selfeat = trainData[selectFeatures]
# impute age with average value
trainData_selfeat = imputeWithAvgValue(trainData_selfeat)
# impute the Embarked feature
trainData_selfeat.Embarked.fillna(value = 'X', inplace=True)
# one hot encode categorical values Embarked and Sex
trainData_ohe = ohe_categorical_data(trainData_selfeat, ['Embarked', 'Sex'], prefix=['Embarked', 'Sex'])

# split the trainData_ohe into training and test sets
seed = 42
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(trainData_ohe, trainData.Survived, test_size=0.2)

### build neural network using tensorflow ###
# placeholder nodes from feature matrix (X) and output (y)
X = tf.placeholder(tf.float32, shape=(None, input_dim), name='X')
y = tf.placeholder(tf.float32, shape=(None), name='y')

# specifiy input, hidden layer neuron number and output layer of the neural network
# To do: nodes in layers is random, need to do hyperparameter optimization
input_dim = X_train.shape[1]
hidden_lyr_1 = 50
hidden_lyr_2 = 50
output_lyr = 1

# shuffle batch data
#def shuffle_batch(X, y, batch_size):
#    rnd_idx = np.random.permutation(len(X))
#    n_batches = len(X) // batch_size
#    for batch_idx in np.array_split(rnd_idx, n_batches):
#        X_batch, y_batch = X[batch_idx], y[batch_idx]
#        yield X_batch, y_batch

# dense NN (dnn)
with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, hidden_lyr_1, name='hidden1', activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, hidden_lyr_2, name='hidden2', activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, output_lyr, name='output', activation=tf.nn.sigmoid)
    
# loss 
with tf.name_scope('loss'):
    binary_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(binary_xentropy, name='loss')
    
# Gradient descent optimizer to minimize the loss 
learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
# check accuracy of the model
with tf.name_scope('eval'):
    correct_pred = tf.equal(tf.round(logits), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
# initialize the tensor graph
init = tf.global_variables_initializer()

# run the graph in session
n_epochs = 20
#batch_size = 1

### accuracy of this model is 58%, need to work on the model 

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        #for X_batch, y_batch in shuffle_batch(X_train, Y_train, batch_size):
            #sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        sess.run(training_op, feed_dict={X:X_train, y:y_train})
        #acc_batch = accuracy.eval(feed_dict={X: X_train, y:Y_train})
        acc_train = accuracy.eval(feed_dict={X: X_train, y:y_train})
        acc_test = accuracy.eval(feed_dict={X: X_test, y:Y_test})
        #print (epoch, 'Batch accuracy: ', acc_batch, 'Test_accuracy: ', acc_test)
        print (epoch, 'Train accuracy: ', acc_train, 'Test_accuracy: ', acc_test)