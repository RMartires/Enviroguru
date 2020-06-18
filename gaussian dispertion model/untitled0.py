# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:13:58 2019

@author: ROHIT
"""
import math
import numpy as np
import matplotlib.pyplot as plt
def conc(x,y,z,hs,z0):

    iy=0.88/(math.log((hs/z0),math.e)-1)
    iz = 0.50 / (math.log((hs / z0), math.e) - 1)

    sigmay=x*iy
    sigmaz=x*iz

    temp1=(Q/U)*(1/(2*math.pi*sigmay*sigmaz))
    temp2=math.exp(-math.pow(y,2)/(2*(math.pow(sigmay,2))))
    temp3=(math.exp((-math.pow((z-H),2))/(2*math.pow(sigmaz,2))) + math.exp((-math.pow((z+H),2)/(2*math.pow(sigmaz,2)))))
    final=temp1*temp2*temp3

    return final;


#iy=0.10
#iz=0.05

Q=2*math.pow(10,5)
hs=120
deltah=10
U=15
z0=0.01
H=hs+deltah
sizex=2*H;
arr=np.zeros((sizex,800))
color=np.zeros((sizex,800),dtype='int32')

for i in range(1,sizex):
    for j in range(1,800):
        arr[i][j]=conc(j,0,i,hs,z0)
        
l1=333397.0790295258566
l2=33333.97079029525857
l3=8333.492697573814642
l4=244.1231743934536604
l5=1.9436330637602476
l6=0.6932814371117969
l7=0.0555
l8=0.005


for i in range(1,2*H):
    for j in range(1,800):
        if arr[i][j]<l8:
            color[i][j]=20
        elif arr[i][j]>l8 and arr[i][j]<l7:
            color[i][j]=50
        elif arr[i][j]>l7 and arr[i][j]<l6:
            color[i][j]=84
        elif arr[i][j]>l6 and arr[i][j]<l5:
            color[i][j]=126
        elif arr[i][j]>l5 and arr[i][j]<l4:
            color[i][j]=168
        elif arr[i][j]>l4 and arr[i][j]<l3:
            color[i][j]=198
        elif arr[i][j]>l3 and arr[i][j]<l2:
            color[i][j]=230
        elif arr[i][j]<l1:
            color[i][j]=255

#white max conc.

print(color)

plt.imshow(color, cmap='gray', vmin=0, vmax=255)
plt.title("Gaussian air dispersion model")
plt.xlabel("distance m")
plt.ylabel("height m")