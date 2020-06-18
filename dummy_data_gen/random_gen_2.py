# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 04:02:45 2019

@author: ROHIT
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#const-def
poll=[50,70] #mean=60 sd=10
date_day=[1,30]
date_month=[1,12]
date_yr=[2017,2018,2019,2020]
Q=[230,270]
output=[60-90]


#rows and cols
features=14
n=5000
X=np.zeros((n,features))


#features= date[11,12,13]=3,Q[0]=1,sizeofindustry[1,2,3,4,5]=1 or 5,typeofproduct[6,7,8,9,10]=1 or 5


#date stuff
day=1
month=1
yr=2017

season=['rain','summer','winter']
SSsize=[1,2,3,4,5]
typeofproduct=['A','B','C','D','E']

#cal slope for maping
maxT=2160
minT=16
maxQ=300
minQ=150
m=(maxQ-minQ)/(maxT-minT)
# output = output_start + m * (input - input_start)


for i in range(n):
    X[i,3]=np.random.normal(70,10)             #output of the factory
    output_temp=X[i,3]
    
    X[i,11]=day                    #day
    
    X[i,12]=month                  #month
    
    X[i,13]=yr                   #year
    day+=1
    if(day > 31):
        day=1
        month+=1
        if(month > 12):
            month=1
            yr+=1
            
#for size of the industry i.e classified into 5 classes
    sssizetemp=np.random.randint(SSsize[0],SSsize[4]+1)
    if(sssizetemp==1 or sssizetemp==2):
        token2=2
    elif(sssizetemp==3 or sssizetemp==4):
        token2=4
    else:
        token2=6
    X[i,1]=sssizetemp
            

#for type of product that it produces
    toptemp=np.random.randint(1,6)
    if(toptemp==1 or toptemp==2):
        token4=2
    elif(toptemp==3 or toptemp==4):
        token4=4
    else:
        token4=6
    X[i,2]=toptemp

            
    #cal token for Q
    if(month>=3 and month <=5):
        season_temp='summer'
        token1=2
    elif(month>=6 and month <=9):
        season_temp='rain'
        token1=4
    else:
        season_temp='winter'
        token1=6
    
            
    if(output_temp>40 and output_temp<60):
        token3=2
    elif(output_temp>60 and output_temp<70):
        token3=4
    elif(output_temp>70 and output_temp<80):
        token3=6
    elif(output_temp>80):
        token3=8
    else:
        token3=10
        
    final_token=token1*token2*token3
    
    Q_mean= minQ + (m*(final_token-minT)) #mapping Qmean
      
    X[i,0]=np.random.normal(Q_mean,10) #final Q
    
            
cols=["Q","soi[0]","soi[1]","soi[2]","soi[3]","soi[4]","top[0]","top[1]","top[2]","top[3]","top[4]","day","month","year"]
#cols=["Q","soi[0]","top[0]","day","month","year"]

pd.DataFrame(X,columns=cols).to_csv("internal_features.csv")
    