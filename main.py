import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk
from tkinter import *

def featureScaling(X):
    for i in range(X.shape[1]):
        if i==0:
            continue;
        X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))
    return X
def implemet():
    X_Temp = np.zeros((150, 2), dtype=float)
    Y = np.zeros((50, 1), dtype=int)
    #get Features we need (columns)
    #feature1
    if(firstFeature.get()=='bill_length_mm'):
       X_Temp[:,0:1]=X_Temp[:,0:1]+data[:,1:2]
    elif(firstFeature.get()=='bill_depth_mm'):
       X_Temp[:,0:1]=X_Temp[:,0:1]+data[:, 2:3]
    elif(firstFeature.get()=='flipper_length_mm'):
       X_Temp[:,0:1]=X_Temp[:,0:1]+data[:, 3:4]
    elif(firstFeature.get()=='gender'):
        X_Temp[:,0:1] = X_Temp[:, 0:1] + data[:, 4:5]
    elif(firstFeature.get()=='body_mass_g'):
        X_Temp[:,0:1] = X_Temp[:, 0:1] + data[:, 5:]

    #feature2
    if(secondFeature.get()=='bill_length_mm'):
       X_Temp[:,1:2]=X_Temp[:,1:2]+data[:,1:2]
    elif(secondFeature.get()=='bill_depth_mm'):
       X_Temp[:,1:2]=X_Temp[:,1:2]+data[:, 2:3]
    elif(secondFeature.get()=='flipper_length_mm'):
       X_Temp[:,1:2]=X_Temp[:,1:2]+data[:, 3:4]
    elif(secondFeature.get()=='gender'):
        X_Temp[:,1:2] = X_Temp[:, 1:2] + data[:, 4:5]
    elif(secondFeature.get()=='body_mass_g'):
        X_Temp[:,1:2] = X_Temp[:, 1:2] + data[:, 5:]
    X=np.zeros((100,2),dtype=float)
    #get Classes we need (rows)
    #class1
    if(firstClass.get()=='Adelie'):
       X[0:50,:]=X_Temp[0:50,:]
    elif(firstClass.get()=='Gentoo'):
       X[0:50,:]=X_Temp[50:100,:]
    elif(firstClass.get()=='Chinstrap'):
       X[0:50,:]=X_Temp[100:150, :]
    #class2
    if(secondClass.get()=='Adelie'):
       X[50:100,:]+=X_Temp[0:50,:]
    elif(secondClass.get()=='Gentoo'):
       X[50:100,:]+=X_Temp[50:100,:]
    elif(secondClass.get()=='Chinstrap'):
       X[50:100,:]+=X_Temp[100:150, :]
    X=X.reshape(100, 2)
    if bias.get()==0: #mfii4
        biasColumn = np.zeros((100,))
    elif bias.get()==1:
        biasColumn = np.ones((100,))
    X = np.insert(X, 0, biasColumn, axis=1)
    X = X.reshape(100, 3)
    #scalling for two features
    #the first 50 sample equal 0 and esle equal 1
    Y= np.append(Y,np.ones((50,1),dtype=int))
    Y[Y==0]=-1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
    plt.scatter(X_train[:,1],X_train[:,2])
    #a4t8alt bias awl  2 features initially l7d ma n3ml al gui
    W=np.random.rand(2,1)
    if bias.get()==0:
     W=np.insert(W,0,0)
    elif bias.get()==1:
     W=np.insert(W,0,1)
    W=W.reshape(3,1)
    #inputs
    #GUI
    c=0
    for j in range(int(iteration.get())):
     for i in range(X_train.shape[0]):
       net = np.dot(X_train[i:i+1,:], W)
       net[net >= 0] = 1
       net[net < 0] = -1
       error= Y_train[i:i+1]-net[0:1,0:1]
       W = W +  error * float(learningRate.get()) * (X_train[i:i+1,:].T)
    net=np.dot(X_test[:,:3],W)
    net[net >= 0] = 1
    net[net < 0] = -1
    for i in range(X_test.shape[0]):
        if Y_test[i]-net[i] == 0:
            c += 1
    accuracy=c/X_test.shape[0]*100
    print("Class 1: ", firstClass.get())
    print("Class 2: ", secondClass.get())
    print("Feature 1: ", firstFeature.get())
    print("Feature 2: ", secondFeature.get())
    print("No of iterations: ",iteration.get())
    print("Learning rate: ",learningRate.get())
    if bias.get()==1:
        print("Bias is included.")
    elif bias.get()==0:
        print("Bias is execluded.")
    print("Accuracy of the System: ",accuracy,"%\n")
    plot_x = np.array([min(X_train[:,0]) - 2, max(X_train[:,0]) + 2])
    plot_y = (-1/W[2]) * (W[1] * plot_x+W[0] )
    plt.plot(plot_x, plot_y, label = "Decision_Boundary")
    plt.show()
    X_Temp=None
    X=None
    Y=None


def run():
    if firstClass.get() == secondClass.get() or firstFeature.get() == secondFeature.get() or learningRate.get() > 1.0:
        Error = Label(window, text="ERROR")
        Error.grid(column=0, row=12, padx=10, pady=25)
    else:
        implemet()




firstClass=None
secondClass=None
firstFeature=None
secondFeature=None
learningRate=None
iteration=None
bias=None

data = pd.read_csv("penguins.csv")
# null bit7wel to male initially
gender = {'male': 1, 'female': 0, np.nan: np.random.randint(0, 2)}  # 2 is execluded
data['gender'] = data['gender'].map(gender)
data = np.array(data)
data= featureScaling(data)

figure, axis=plt.subplots(3,3,figsize=(15,25))

listData=["bill_length_mm", "bill_depth_mm","flipper_length_mm","gender","body_mass_g"]
counterX=-1
counterY=0
for i in range(len(listData)):
    for j in range(i+1,len(listData)):
        if (counterX == 2 and counterY == 2):
            break;
        counterX+=1
        if (counterX == 3):
            counterY += 1
            counterX = 0
        axis[counterX, counterY].scatter(data[:50, i+1:i+2], data[:50, j+1:j+2], color='red')
        axis[counterX, counterY].scatter(data[50:100, i+1:i+2], data[50:100, j+1:j+2], color='green')
        axis[counterX, counterY].scatter(data[100:150, i+1:i+2], data[100:150, j+1:j+2], color='black')
        str=listData[i]," and ", listData[j]
        axis[counterX, counterY].set_title(str)
plt.show()

# check box of class [1,2,3], check box of class2[1,2,3]

window = tk.Tk()
window.title('Class')
window.geometry('500x700')

ttk.Label(window, text = "Select First Class :",font = ("Times New Roman", 10)).grid(column = 0,row = 5, padx = 10, pady = 25)
ttk.Label(window, text = "Select Second Class :",font = ("Times New Roman", 10)).grid(column = 0,row = 6, padx = 10, pady = 25)
# Combobox creation
firstClass = tk.StringVar()
secondClass=  tk.StringVar()
Class1 = ttk.Combobox(window, width = 27, textvariable = firstClass)
Class2 = ttk.Combobox(window, width = 27, textvariable = secondClass)
# Adding combobox drop down list
Class1['values'] = ('Adelie','Gentoo','Chinstrap',)
Class2['values'] = ('Adelie','Gentoo','Chinstrap',)
Class1.grid(column = 1, row = 5)
Class2.grid(column = 1, row = 6)
Class1.current()
Class2.current()
firstFeature= tk.StringVar()
secondFeature=tk.StringVar()
feature1 = ttk.Combobox(window, width = 27, textvariable = firstFeature)
feature2 = ttk.Combobox(window, width = 27, textvariable = secondFeature)
# Adding combobox drop down list
feature1['values'] = ('bill_length_mm','bill_depth_mm','flipper_length_mm','gender','body_mass_g')
feature2['values'] = ('bill_length_mm','bill_depth_mm','flipper_length_mm','gender','body_mass_g')
feature1.grid(column = 1,row = 7, padx = 10, pady = 25)
feature2.grid(column = 1,row = 8, padx = 10, pady = 25)
ttk.Label(window, text = "Select First Feature :",font = ("Times New Roman", 10)).grid(column = 0,row = 7, padx = 10, pady = 25)
ttk.Label(window, text = "Select Second Feature :",font = ("Times New Roman", 10)).grid(column = 0,row = 8, padx = 10, pady = 25)
feature1.current()
feature2.current()
Label1 = Label(window,text="Learning Rate").grid(column = 0,row = 9, padx = 10, pady = 25)
Label2 = Label(window,text="echo").grid(column = 0,row = 10, padx = 10, pady = 25)
learningRate = tk.DoubleVar()
iteration=tk.IntVar()
bias = tk.IntVar()
k= tk.Checkbutton(window, text='Bias',variable=bias, onvalue=1, offvalue=0)
k.grid(column = 1,row = 11, padx = 10, pady = 25)
LR = Entry(window,width=30, textvariable = learningRate)
LR.grid(column = 1,row = 9, padx = 10, pady = 25)
LR.focus_set()
echo = Entry(window, width=30,textvariable=iteration)
echo.grid(column = 1,row = 10, padx = 10, pady = 25)
Run = Button(window,text="Run",command=run).grid(column = 1,row = 13, padx = 10, pady = 25)

window.mainloop()
