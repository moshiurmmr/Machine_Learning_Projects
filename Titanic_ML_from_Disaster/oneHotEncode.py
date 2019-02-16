#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 22:43:27 2019

@author: moshiur
"""
import pandas as pd

# function for One  Hot Encoding the categorical features in the column of the data
def ohe_categorical_data(data, columns, prefix):
    data = pd.get_dummies(data, columns=columns, prefix=prefix)
    return data 