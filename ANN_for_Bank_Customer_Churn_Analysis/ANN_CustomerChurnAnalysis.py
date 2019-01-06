#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 19:32:54 2018

@author: moshiur
"""

# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf



# activation function leaky_relu
"""
def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)
"""
    
# ANN graph
n_inputs = 11 # 10 inputs + 1 bias 
n_hidden = 6 # arbitary, talking average of i/p (11) and o/p (1) layer nodes
#n_hidden = 10 # arbitrary, n_input - 1
n_outputs = 1 # binary output

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
#y = tf.placeholder(tf.int32, shape=(None), name='y')
y = tf.placeholder(tf.float32, shape=(None), name='y')
# DNN with leaky_relu activation function at the o/p layer
initializer = tf.contrib.layers.xavier_initializer()
with tf.name_scope('dnn'):
#    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.relu, name='hidden', 
#                             kernel_initializer=initializer)
    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.relu, name='hidden') # no kernel initializer
    logits = tf.layers.dense(hidden, n_outputs, activation=None,
                             name='outputs')
    
# DNN with sigmoid in the o/p layer
#with tf.name_scope('dnn'):
#    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.sigmoid, name='hidden')
#    logits = tf.layers.dense(hidden, n_outputs, name='outputs')

# cost function: loss
with tf.name_scope('loss'):
    binaryxentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(binaryxentropy, name='loss')
#with tf.name_scope('loss'):
#    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#    loss = tf.reduce_mean(xentropy, name='loss')


# optimize loss
learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

# evaluate 
#with tf.name_scope('eval'):
#    #correct = tf.nn.
#    #correct = tf.nn.in_top_k(logits, y, 1)
#    #print(tf.cast(y, tf.int32))
#    correct = tf.nn.in_top_k(logits, y, 1)
#    #correct = tf.nn.in_top_k(logits, tf.cast(y, tf.int32), 1) 
#    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

predicted = tf.nn.sigmoid(logits)
correct_pred = tf.equal(tf.round(predicted), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# initialize nodes
init = tf.global_variables_initializer()
#saver = tf.train.Saver()

# import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# columns from 3 to 12 as the independent variables
X_dataset = dataset.iloc[:, 3:13].values
# binary dependent variable denoting whether customer exited the bank or not - col-13
y_dataset = dataset.iloc[:, 13].values
# transform categorical variable 'Geograpy - X[:, 1] 
labelencoder_X_1 = LabelEncoder()
X_dataset[:, 1] = labelencoder_X_1.fit_transform(X_dataset[:, 1])
# transform categorical variable  Gender - X[:, 2]
labelencoder_X_2 = LabelEncoder()
X_dataset[:, 2] = labelencoder_X_1.fit_transform(X_dataset[:, 2])
# apply OneHotEncoder to only X[:, 1]. X[:, 2] doesn't need one-hot-encoding as it is binary
onehotencoder = OneHotEncoder(categorical_features=[1])
X_dataset = onehotencoder.fit_transform(X_dataset).toarray()
# remove the first column of the one-hot-encoded 'Geography' variable to remove 'dummy variable trap'
X_dataset = X_dataset[:, 1:]

# split the data to training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size = 0.2, random_state = 42)

## converting data types
#X_train = X_train.astype(np.float32)
#X_test = X_test.astype(np.float32)
##y_train = y_train.astype(np.int32)
##y_test = y_test.astype(np.int32)
#y_train = y_train.astype(np.float32)
#y_test = y_test.astype(np.float32)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

#sc_y = StandardScaler()
#y_train = (sc_y.fit_transform(y_train)).reshape(-1, 1)
#y_test = sc_y.transform(y_test)
# validation and training sets
X_train, X_val = X_train[0:6000, :], X_train[6000:,]
y_train, y_val = y_train[0:6000], y_train[6000:]

# function to shuffle the indices and create training batches 
def shuffle_batch(X, y, batch_size):
    random_index = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_ind in np.array_split(random_index, n_batches):
        X_batch, y_batch = X[batch_ind], y[batch_ind]
        yield X_batch, y_batch

# train the model in tf.session
n_epochs = 40
batch_size = 50

with tf.Session() as sess:
    init.run() # initialize variables
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
           
        if epoch % 5 == 0:
            #print('testing epochs')
            loss_val, _, acc_val = sess.run([loss, training_op, accuracy], 
                                         feed_dict={X:X_val, y:y_val})
            print('Epoch: {:5}\t Loss: {:.3f}\t Accuracy: {:.2%}'.format(epoch, loss_val, 
                  acc_val))
    print('Test accuracy: ', sess.run(accuracy,feed_dict={X:X_test, y:y_test}))
        #acc_batch = accuracy.eval(feed_dict{X: X_batch, y:y_batch})
#            acc_batch = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
#            acc_valid = accuracy.eval(feed_dict = {X:X_val, y:y_val})
#            print(epoch, 'Batch accuracy: ', acc_batch,
#                       'Validation accuracy: ', acc_valid)
    #save_path = saver.save(sess, './BankCustomerChurnAnalysis.ckpt')
    