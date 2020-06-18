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


kr=['linear', 'rbf', 'sigmoid']
deg=[1,2,3,4,5]

# Fitting SVR to the dataset
from sklearn.svm import SVR
for j in kr:

    regressor = SVR(kernel = j)
    regressor.fit(X_train, y_train)
    tabel_test=np.zeros((len(X_test),2))
        
    for i in range(len(X_test)):
        y_pred = regressor.predict(X_test[i].reshape(1,-1))
        tabel_test[i,0]=y_pred
        tabel_test[i,1]=y_test[i]
         
    rmstest = sqrt(mean_squared_error(tabel_test[:,1], tabel_test[:,0]))
    rms_test.append(rmstest)
         
       
    tabel_train=np.zeros((len(X_test),2))
    for i in range(len(X_test)):
        y_pred = regressor.predict(X_train[i].reshape(1,-1))
        tabel_train[i,0]=y_pred
        tabel_train[i,1]=y_train[i]
         
    rmstrain = sqrt(mean_squared_error(tabel_train[:,1], tabel_train[:,0]))
    rms_train.append(rmstrain)
    print("d")



plt.plot([1,2,3],rms_train, color = 'blue',label="train")
plt.plot([1,2,3],rms_test, color = 'red',label="test")
plt.legend(loc="center right")
plt.title('Kernel (SVR)')
plt.xlabel('kernel [linear,rbf, sigmoid,]')
plt.ylabel('RMS error m\u00b3/hr')
plt.show()
