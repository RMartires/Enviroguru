# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:01:44 2019

@author: ROHIT
"""

# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error
from math import sqrt

# Importing the dataset
dataset = pd.read_csv('test5.csv')
X = dataset.iloc[:,1:5].values
y = dataset.iloc[:,0].values
#X = X.reshape(-1,1);
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

rms_test=[]
rms_train=[]


# Fitting SVR to the dataset
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
tabel_test=np.zeros((40,2))
    
for i in range(40):
    y_pred = regressor.predict(X_test[i].reshape(1,-1))
    tabel_test[i,0]=y_pred
    tabel_test[i,1]=y_test[i]
     
rmstest = sqrt(mean_squared_error(tabel_test[:,1], tabel_test[:,0]))
rms_test.append(rmstest)
     
   
tabel_train=np.zeros((40,2))
for i in range(40):
    y_pred = regressor.predict(X_train[i].reshape(1,-1))
    tabel_train[i,0]=y_pred
    tabel_train[i,1]=y_train[i]
     
rmstrain = sqrt(mean_squared_error(tabel_train[:,1], tabel_train[:,0]))
rms_train.append(rmstrain)


plt.plot(range(0,len(tabel_train[:,1])),tabel_train[:,0], color = 'blue',label="prediction")
plt.plot(range(0,len(tabel_train[:,1])),tabel_train[:,1], color = 'red',label="actual")
plt.legend(loc="top right")
plt.title('multilinear regression')
plt.xlabel('index of saples from train data')
plt.ylabel('emission rate m\u00b3/hr')
plt.show()
