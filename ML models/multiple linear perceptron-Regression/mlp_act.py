# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:56:29 2019

@author: ROHIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.neural_network import MLPRegressor


dataset = pd.read_csv('test.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 0].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


hd=[10,20,50,100,150,200,300,400,500]
act=['identity', 'logistic', 'tanh', 'relu']

solver=['lbfgs', 'sgd', 'adam']


rms_test=[]
rms_train=[]

for j in act:
    mlp=MLPRegressor(hidden_layer_sizes=300,activation=j)
    mlp.fit(X_train,y_train)
    
    
    tabel_test=np.zeros((len(X_test),2))
        
    for i in range(len(X_test)):
        y_pred = mlp.predict(X_test[i].reshape(1,-1))
        tabel_test[i,0]=y_pred
        tabel_test[i,1]=y_test[i]
         
    rmstest = sqrt(mean_squared_error(tabel_test[:,1], tabel_test[:,0]))
    rms_test.append(rmstest)
         
       
    tabel_train=np.zeros((len(X_test),2))
    for i in range(len(X_test)):
        y_pred = mlp.predict(X_train[i].reshape(1,-1))
        tabel_train[i,0]=y_pred
        tabel_train[i,1]=y_train[i]
         
    rmstrain = sqrt(mean_squared_error(tabel_train[:,1], tabel_train[:,0]))
    rms_train.append(rmstrain)

print(mlp.score(X_test,y_test))

plt.plot([1,2,3,4],rms_train, color = 'blue',label="test")
plt.plot([1,2,3,4],rms_test, color = 'red',label="train")
plt.legend(loc="center right")
plt.title('activation function (MLP-regression)')
plt.xlabel('activation function [identity logistic tanh  relu])')
plt.ylabel('RMS error m\u00b3/hr')
plt.show()
