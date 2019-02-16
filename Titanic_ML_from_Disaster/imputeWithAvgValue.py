#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 22:01:42 2019

@author: moshiur
"""
### need to work on this function to make it more generic ###

#def imputeWithAvgValue(data, features):
#    for feature in features:
#        data.feature.fillna(value = data.feature.mean(), inplace=True)
#    return data

def imputeWithAvgValue(data):
        data.Age.fillna(value = data.Age.mean(), inplace=True)
        return data