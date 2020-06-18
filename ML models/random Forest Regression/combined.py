# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 00:25:39 2019

@author: ROHIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
#tuning files
import min_sample_split as mss
import EG_rf as nest
import maxdepth_rf as md
import maxfeatures as mf
import min_sample_leaf as msl


# Importing the dataset
dataset = pd.read_csv('test5.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)






mss.min_sample_split(X_train, X_test, y_train, y_test)
nest.N_est(X_train, X_test, y_train, y_test)
md.max_depth(X_train, X_test, y_train, y_test)
mf.maxfeatures(X_train, X_test, y_train, y_test)
msl.min_sample_leaf(X_train, X_test, y_train, y_test)


