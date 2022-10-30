import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk
from tkinter import *

c1=None
c2=None
f1=None
f2=None
l_s=None
iter=None
b=None

def w():
    data = pd.read_csv("penguins.csv")
    # null bit7wel to male initially
    gender = {'male': 1, 'female': 0, np.nan:np.random.randint(0,2) } #2 is execluded
    data['gender'] = data['gender'].map(gender)
    data= np.array(data)
    # check box of class [1,2,3], check box of class2[1,2,3]
    X_befor = np.zeros((150, 2), dtype=float)
    Y = np.zeros((50, 1), dtype=int)
    if(f1.get()=='bill_length_mm'):
       X_befor[:,0:1]=X_befor[:,0:1]+data[:,1:2]
       X_befor[:, 0:1] = (X_befor[:, 0:1] - np.min(X_befor[:, 0:1])) / (np.max(X_befor[:, 0:1]) - np.min(X_befor[:, 0:1]))
    elif(f1.get()=='bill_depth_mm'):
       X_befor[:,0:1]=X_befor[:,0:1]+data[:, 2:3]
       X_befor[:, 0:1] = (X_befor[:, 0:1] - np.min(X_befor[:, 0:1])) / (np.max(X_befor[:, 0:1]) - np.min(X_befor[:, 0:1]))
    elif(f1.get()=='flipper_length_mm'):
       X_befor[:,0:1]=X_befor[:,0:1]+data[:, 3:4]
       X_befor[:, 0:1] = (X_befor[:, 0:1] - np.min(X_befor[:, 0:1])) / (np.max(X_befor[:, 0:1]) - np.min(X_befor[:, 0:1]))
    elif(f1.get()=='gender'):
        X_befor[:,0:1] = X_befor[:, 0:1] + data[:, 4:5]
    elif(f1.get()=='body_mass_g'):
        X_befor[:,0:1] = X_befor[:, 0:1] + data[:, 5:]
        X_befor[:, 0:1] = (X_befor[:, 0:1] - np.min(X_befor[:, 0:1])) / (np.max(X_befor[:, 0:1]) - np.min(X_befor[:, 0:1]))
    if(f2.get()=='bill_length_mm'):
       X_befor[:,1:2]=X_befor[:,1:2]+data[:,1:2]
       X_befor[:, 1:2] = (X_befor[:, 1:2] - np.min(X_befor[:, 1:2])) / (np.max(X_befor[:, 1:2]) - np.min(X_befor[:, 1:2]))
    elif(f2.get()=='bill_depth_mm'):
       X_befor[:,1:2]=X_befor[:,1:2]+data[:, 2:3]
       X_befor[:, 1:2] = (X_befor[:, 1:2] - np.min(X_befor[:, 1:2])) / (np.max(X_befor[:, 1:2]) - np.min(X_befor[:, 1:2]))
    elif(f2.get()=='flipper_length_mm'):
       X_befor[:,1:2]=X_befor[:,1:2]+data[:, 3:4]
       X_befor[:, 1:2] = (X_befor[:, 1:2] - np.min(X_befor[:, 1:2])) / (np.max(X_befor[:, 1:2]) - np.min(X_befor[:, 1:2]))
    elif(f2.get()=='gender'):
        X_befor[:,1:2] = X_befor[:, 1:2] + data[:, 4:5]
    elif(f2.get()=='body_mass_g'):
        X_befor[:,1:2] = X_befor[:, 1:2] + data[:, 5:]
        X_befor[:, 1:2] = (X_befor[:, 1:2] - np.min(X_befor[:, 1:2])) / (np.max(X_befor[:, 1:2]) - np.min(X_befor[:, 1:2]))
    X=np.zeros((100,2),dtype=float)
    if(c1.get()=='Adelie'):
       X[0:50,:]=X_befor[0:50,:]
    elif(c1.get()=='Gentoo'):
       X[0:50,:]=X_befor[50:100,:]
    elif(c1.get()=='Chinstrap'):
       X[0:50,:]=X_befor[100:150, :]
    if(c2.get()=='Adelie'):
       X[50:100,:]+=X_befor[0:50,:]
    elif(c2.get()=='Gentoo'):
       X[50:100,:]+=X_befor[50:100,:]
    elif(c2.get()=='Chinstrap'):
       X[50:100,:]+=X_befor[100:150, :]
    X=X.reshape(100, 2)
    if b.get()==0: #mfii4
        bia = np.zeros((100,))
    elif b.get()==1:
        bia = np.ones((100,))
    X = np.insert(X, 0, bia, axis=1)
    X = X.reshape(100, 3)
    #scalling for two features
    #the first 50 sample equal 0 and esle equal 1
    Y= np.append(Y,np.ones((50,1),dtype=int))
    Y[Y==0]=-1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
    plt.scatter(X_train[:,1],X_train[:,2])
    #a4t8alt b awl  2 features initially l7d ma n3ml al gui
    W=np.random.rand(2,1)
    if b.get()==0:
     W=np.insert(W,0,0)
    elif b.get()==1:
     W=np.insert(W,0,1)
    W=W.reshape(3,1)
    #inputs
    #GUI
    c=0
    for j in range(int(iter.get())):
     for i in range(X_train.shape[0]):
       Z = np.dot(X_train[i:i+1,:], W)
       Z[Z >= 0] = 1
       Z[Z < 0] = -1
       e= Y_train[i:i+1]-Z[0:1,0:1]
       W = W +  e * float(l_s.get()) * (X_train[i:i+1,:].T)
    Z=np.dot(X_test[:,:3],W)
    Z[Z >= 0] = 1
    Z[Z < 0] = -1
    cost=0
    for i in range(X_test.shape[0]):
        if Y_test[i]-Z[i] == 0:
            c += 1
    accuracy=c/X_test.shape[0]*100
    print("Class 1: ", c1.get())
    print("Class 2: ", c2.get())
    print("Feature 1: ", f1.get())
    print("Feature 2: ", f2.get())
    print("No of iterations: ",iter.get())
    print("Learning rate: ",l_s.get())
    if b.get()==1:
        print("Bias is included.")
    elif b.get()==0:
        print("Bias is execluded.")
    print("Accuracy of the System: ",accuracy,"%\n")
    plot_x = np.array([min(X_train[:,0]) - 2, max(X_train[:,0]) + 2])
    plot_y = (-1/W[2]) * (W[1] * plot_x+W[0] )
    plt.plot(plot_x, plot_y, label = "Decision_Boundary")
    plt.show()
    X_befor=None
    X=None
    Y=None
window = tk.Tk()
window.title('Class')
window.geometry('500x700')
def r():

     if c1.get() == c2.get() or f1.get() == f2.get() or l_s.get()>1.0:
         Error = Label(window,text="ERROR")
         Error.grid(column=0,row=12, padx=10, pady=25)
     else:
         w()

ttk.Label(window, text = "Select First Class :",font = ("Times New Roman", 10)).grid(column = 0,row = 5, padx = 10, pady = 25)
ttk.Label(window, text = "Select Second Class :",font = ("Times New Roman", 10)).grid(column = 0,row = 6, padx = 10, pady = 25)
# Combobox creation
c1 = tk.StringVar()
c2=  tk.StringVar()
Class1 = ttk.Combobox(window, width = 27, textvariable = c1)
Class2 = ttk.Combobox(window, width = 27, textvariable = c2)
# Adding combobox drop down list
Class1['values'] = ('Adelie','Gentoo','Chinstrap',)
Class2['values'] = ('Adelie','Gentoo','Chinstrap',)
Class1.grid(column = 1, row = 5)
Class2.grid(column = 1, row = 6)
Class1.current()
Class2.current()
f1= tk.StringVar()
f2=tk.StringVar()
feature1 = ttk.Combobox(window, width = 27, textvariable = f1)
feature2 = ttk.Combobox(window, width = 27, textvariable = f2)
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
l_s = tk.DoubleVar()
iter=tk.IntVar()
b = tk.IntVar()
k= tk.Checkbutton(window, text='Bias',variable=b, onvalue=1, offvalue=0)
k.grid(column = 1,row = 11, padx = 10, pady = 25)
LR = Entry(window,width=30, textvariable = l_s)
LR.grid(column = 1,row = 9, padx = 10, pady = 25)
LR.focus_set()
echo = Entry(window, width=30,textvariable=iter)
echo.grid(column = 1,row = 10, padx = 10, pady = 25)
Run = Button(window,text="Run",command=r).grid(column = 1,row = 13, padx = 10, pady = 25)

window.mainloop()