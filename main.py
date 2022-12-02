import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk
from tkinter import *

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
# check box of class [1,2,3], check box of class2[1,2,3]
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
window = tk.Tk()
window.title('Class')
window.geometry('500x700')
def run():
     if firstClass.get() == secondClass.get() or firstFeature.get() == secondFeature.get() or learningRate.get()>1.0:
         Error = Label(window,text="ERROR")
         Error.grid(column=0,row=12, padx=10, pady=25)
     else:
         implemet()
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
data[:, 1:2] = (data[:, 1:2] - np.min(data[:, 1:2])) / (np.max(data[:, 1:2]) - np.min(data[:, 1:2]))
data[:, 2:3] = (data[:, 2:3] - np.min(data[:, 2:3])) / (np.max(data[:, 2:3]) - np.min(data[:, 2:3]))
data[:, 3:4] = (data[:, 3:4] - np.min(data[:, 3:4])) / (np.max(data[:, 3:4]) - np.min(data[:, 3:4]))
data[:, 5:] = (data[:, 5:] - np.min(data[:, 5:])) / (np.max(data[:, 5:]) - np.min(data[:, 5:]))

figure, axis=plt.subplots(3,3,figsize=(15,25))
axis[0,0].scatter(data[:50,1:2], data[:50,2:3],color='green')
axis[0,0].scatter(data[50:100,1:2],data[50:100,2:3],color='red')
axis[0,0].scatter(data[100:150,1:2],data[100:150,2:3],color='black')
axis[0,0].set_title("bill_length_mm and bill_depth_mm")

axis[1,0].scatter(data[:50,1:2], data[:50,3:4],color='green')
axis[1,0].scatter(data[50:100,1:2],data[50:100,3:4],color='red')
axis[1,0].scatter(data[100:150,1:2],data[100:150,3:4],color='black')
axis[1,0].set_title("bill_length_mm and flipper_length_mm")
axis[2,0].scatter(data[:50,1:2], data[:50,4:5],color='green')
axis[2,0].scatter(data[50:100,1:2],data[50:100,4:5],color='red')
axis[2,0].scatter(data[100:150,1:2],data[100:150,4:5],color='black')
axis[2,0].set_title("bill_length_mm and gender")
axis[0,2].scatter(data[:50,1:2], data[:50,5:],color='green')
axis[0,2].scatter(data[50:100,1:2],data[50:100,5:],color='red')
axis[0,2].scatter(data[100:150,1:2],data[100:150,5:],color='black')
axis[0,2].set_title("bill_length_mm and body_mass_g")

axis[0,1].scatter(data[:50,2:3], data[:50,3:4],color='green')
axis[0,1].scatter(data[50:100,2:3],data[50:100,3:4],color='red')
axis[0,1].scatter(data[100:150,2:3],data[100:150,3:4],color='black')
axis[0,1].set_title("bill_depth_mm and flipper_length_mm")

axis[1,1].scatter(data[:50,2:3], data[:50,4:5],color='green')
axis[1,1].scatter(data[50:100,2:3],data[50:100,4:5],color='red')
axis[1,1].scatter(data[100:150,2:3],data[100:150,4:5],color='black')
axis[1,1].set_title("bill_depth_mm and gender")

axis[2,1].scatter(data[:50,2:3], data[:50,4:5],color='green')
axis[2,1].scatter(data[50:100,2:3],data[50:100,4:5],color='red')
axis[2,1].scatter(data[100:150,2:3],data[100:150,4:5],color='black')
axis[2,1].set_title("bill_depth_mm and gender")

axis[1,2].scatter(data[:50,3:4], data[:50,4:5],color='green')
axis[1,2].scatter(data[50:100,3:4],data[50:100,4:5],color='red')
axis[1,2].scatter(data[100:150,3:4],data[100:150,4:5],color='black')
axis[1,2].set_title("flipper_length_mm and gender")

axis[2,2].scatter(data[:50,3:4], data[:50,5:],color='green')
axis[2,2].scatter(data[50:100,3:4],data[50:100,5:],color='red')
axis[2,2].scatter(data[100:150,3:4],data[100:150,5:],color='black')
axis[2,2].set_title("flipper_length_mm and body_mass_g")







plt.show()

window.mainloop()
