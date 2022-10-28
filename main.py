import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split


def featureScaling(X, a, b):
   X = np.array(X)
   Normalized_X = np.zeros((X.shape[0], X.shape[1]))
   for i in range(X.shape[1]):
      Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
   return Normalized_X
f1=1
f2=2


d =pd.read_csv("penguins.csv")
gender ={'male':1,'female':0,np.nan:1}

d['gender']=d['gender'].map(gender)
d =np.array(d)

data=d
#check box of class [0,1,2], check box of class2[0,1,2]
no1 =1;no2=3
X=np.zeros((50,5),dtype=int)
Y=np.zeros((50,1),dtype=int)

if(no1==1):
   X=X+data[0:50,1:6]
elif(no1==2):
   X=X+data[50:100, 1:6]
elif(no1==3):
   X=X+data[100:150, 1:6]
if(no2==1):
   X=X+ np.append(X,data[0:50,1:6])
elif(no2==2):
   X=X+ np.append(X, data[50:100, 1:6])
elif(no2==3):
   X= np.append(X, data[100:150, 1:6])
X=X.reshape(100, 5)
X[:,2] = (X[:,2]-np.min(X[:,2]))/(np.max(X[:,2])-np.min(X[:,2]))
X[:,4] = (X[:,4]-np.min(X[:,4]))/(np.max(X[:,4])-np.min(X[:,4]))
Y= np.append(Y,np.ones((50,1),dtype=int))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
X_train=X_train[:,:2]
W=np.random.rand(2,1)
Z= np.dot(X_train,W)
print(Z)


















