#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 19:58:40 2019

@author: moshiur
"""
import pandas as pd

def removeUniqueFeature(data, features):
    for feature in features:
        if data.shape[0] == len(pd.Series(data[feature]).unique()):
            features.remove(feature)
    return features