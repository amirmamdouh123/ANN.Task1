import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tkinter import *

d =pd.read_csv("penguins.csv")
#null bit7wel to male initially
gender ={'male':1,'female':0,np.nan:1}

d['gender']=d['gender'].map(gender)
d =np.array(d)

data=d
#check box of class [1,2,3], check box of class2[1,2,3]
no1 =1;no2=2
X=np.zeros((50,5),dtype=int)
Y=np.zeros((50,1),dtype=int)

if(no1==1):
   X=X+data[0:50,1:6]
elif(no1==2):
   X=X+data[50:100, 1:6]
elif(no1==3):
   X=X+data[100:150, 1:6]
if(no2==1):
   X=np.append(X,data[0:50,1:6])
elif(no2==2):
   X=np.append(X, data[50:100, 1:6])
elif(no2==3):
   X= np.append(X, data[100:150, 1:6])
X=X.reshape(100, 5)
#scalling for two features
X[:,2] = (X[:,2]-np.min(X[:,2]))/(np.max(X[:,2])-np.min(X[:,2]))
X[:,4] = (X[:,4]-np.min(X[:,4]))/(np.max(X[:,4])-np.min(X[:,4]))
#the first 50 sample equal 0 and esle equal 1
Y= np.append(Y,np.ones((50,1),dtype=int))
Y[Y==0]=-1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
plt.scatter(X_train[:,0],X_train[:,1])
#a4t8alt b awl  2 features initially l7d ma n3ml al gui
X_train=X_train[:,:2]

W=np.random.rand(2,1)
#inputs
l_r =0.5
iter=4
#GUI
b=0
for j in range(iter):
 for i in range(X_train.shape[0]):
   Z = np.dot(X_train[i:i+1,:], W)+b
   Z[Z >= 0] = 1
   Z[Z < 0] = -1
   e= Y_train[i]-Z[0,0]
   W = W +  e * l_r * (X_train[i:i+1,:].T)

Z=np.dot(X_test[:,:2],W)
Z[Z >= 0] = 1
Z[Z < 0] = -1
cost=0
for i in range(X_test.shape[0]):
  cost+=(Y_test[i]-Z[i])
print(cost)
plot_x = np.array([min(X_train[:,0]) - 2, max(X_train[:,0]) + 2])
plot_y = (-1/W[1]) * (W[0] * plot_x+1 )
plt.plot(plot_x, plot_y, label = "Decision_Boundary")
plt.show()

