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

# import Keras Sequential model and Dense layer for neural network building
from keras.models import Sequential
from keras.layers import Dense

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
X_train, X_test, Y_train, Y_test = train_test_split(trainData_ohe, trainData.Survived, test_size=0.2)

# specifiy input, hidden layer neuron number and output layer of the neural network
# To do: nodes in layers is random, need to do hyperparameter optimization
input_dim = X_train.shape[1]
hidden_lyr_1 = 50
hidden_lyr_2 = 50
output_lyr = 1

# build the neural network model
nn_model = Sequential()
# input layer
nn_model.add(Dense(hidden_lyr_1, input_dim=input_dim, activation='relu'))
# 1st hidden layer
nn_model.add(Dense(hidden_lyr_1, activation='relu'))
# 2nd hidden layer
nn_model.add(Dense(hidden_lyr_2, activation='relu'))
# output layer
nn_model.add(Dense(1, activation='sigmoid'))
# compile the neural network
nn_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# fit the model with the training data
nn_model.fit(X_train, Y_train,
            batch_size = 20, epochs = 20, verbose = 1,
            validation_data = (X_test, Y_test))