#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 22:32:02 2019

@author: moshiur
"""
import pandas as pd
import numpy as np

# import custom functions
from removeUniqueFeature import removeUniqueFeature
from imputeWithAvgValue import imputeWithAvgValue
from oneHotEncode import ohe_categorical_data

# to build a NN model using Tensorflow
import tensorflow as tf
from sklearn.model_selection import train_test_split

# import classical classification models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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

# create a Logistic regression classification model
LogReg_clf = LogisticRegression()
LogReg_clf.fit(X_train, Y_train)
# predict on test set
predict_lgc = LogReg_clf.predict(X_test)
# accuracy of the LG classifier
accuracy_lgc = accuracy_score(predict_lgc, Y_test)
print('Accuracy of logictic regression model is: {} '.format(accuracy_lgc))


# Classification using Decision Tree classifier
Dt_clf = DecisionTreeClassifier(max_depth=20)
Dt_clf.fit(X_train, Y_train)
predict_dtc = Dt_clf.predict(X_test)
accuracy_dtc = accuracy_score(predict_dtc, Y_test) 
print('Accuracy of Decission Tree Classifier is: {}'.format(accuracy_dtc))

# classification using RadomForest Classifier
Rf_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=16, n_jobs=-1)
Rf_clf.fit(X_train, Y_train)
# predict test score using RFC
predict_rfc = Rf_clf.predict(X_test)
accuracy_rfc = accuracy_score(predict_rfc, Y_test)
print('Accuracy of Random Forest Classifer is: {}'.format(accuracy_rfc))
