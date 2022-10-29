import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import tkinter as tk
from tkinter import ttk
from tkinter import *
window = tk.Tk()
window.title('Class')
window.geometry('500x600')
class1 = None
class2 = None
l_r = None
iter = None
def r():
    if Class1.get()==Class2.get() or feature1.get() ==feature2.get():
        Error = Label(window,
                          text="ERROR").grid(column = 0,
		row = 11, padx = 10, pady = 25)
    else:
     class1=Class1.get()
     class2=Class2.get()
     l_r=user_name_input_area
     iter=user_password_entry_area
     print(class1)

     window.destroy()

# label text for title

# label
ttk.Label(window, text = "Select First Class :",
		font = ("Times New Roman", 10)).grid(column = 0,
		row = 5, padx = 10, pady = 25)
ttk.Label(window, text = "Select Second Class :",
		font = ("Times New Roman", 10)).grid(column = 0,
		row = 6, padx = 10, pady = 25)

# Combobox creation
c1 = tk.StringVar()
c2=tk.StringVar()
Class1 = ttk.Combobox(window, width = 27, textvariable = c1)
Class2 = ttk.Combobox(window, width = 27, textvariable = c2)

# Adding combobox drop down list
Class1['values'] = ('Adelie',
						' Gentoo',
						' Chinstrap',
						)
Class2['values'] = ('Adelie',
						' Gentoo',
						' Chinstrap',
						)

Class1.grid(column = 1, row = 5)
Class2.grid(column = 1, row = 6)

Class1.current()
Class2.current()


f1 = tk.StringVar()
f2=tk.StringVar()
feature1 = ttk.Combobox(window, width = 27, textvariable = f1)
feature2 = ttk.Combobox(window, width = 27, textvariable = f2)

# Adding combobox drop down list
feature1['values'] = ('bill_length_mm',
						'bill_depth_mm',
                    ' flipper_length_mm',
                    ' gender',
                      'body_mass_g'
)
feature2['values'] =('bill_length_mm',
						'bill_depth_mm',
                    ' flipper_length_mm',
                    ' gender',
                      'body_mass_g'
)
feature1.grid(column = 1,
		row = 7, padx = 10, pady = 25)

feature2.grid(column = 1,
		row = 8, padx = 10, pady = 25)

ttk.Label(window, text = "Select First Feature :",
		font = ("Times New Roman", 10)).grid(column = 0,
		row = 7, padx = 10, pady = 25)
ttk.Label(window, text = "Select Second Feature :",
		font = ("Times New Roman", 10)).grid(column = 0,
		row = 8, padx = 10, pady = 25)

feature1.current()
feature2.current()





# the label for user_name
user_name = Label(window,
                  text="Learning Rate").grid(column = 0,
		row = 7, padx = 10, pady = 25)

# the label for user_password
user_password = Label(window,
                      text="echo").grid(column = 0,
		row = 8, padx = 10, pady = 25)


user_name_input_area = Entry(window,
                             width=30).grid(column = 1,
		row = 9, padx = 10, pady = 25)

user_password_entry_area = Entry(window,
                                 width=30).grid(column = 1,
		row = 10, padx = 10, pady = 25)
submit_button = Button(window,
                       text="Run",command=r).grid(column = 1,
		row = 12, padx = 10, pady = 25)


window.mainloop()

print(class1)





d =pd.read_csv("penguins.csv")
#null bit7wel to male initially
gender ={'male':1,'female':0,np.nan:1}

d['gender']=d['gender'].map(gender)
d =np.array(d)

data=d
#check box of class [1,2,3], check box of class2[1,2,3]
X=np.zeros((50,5),dtype=int)
Y=np.zeros((50,1),dtype=int)

if(class1=='Adelie'):
   X=X+data[0:50,1:6]
elif(class1=='Gentoo'):
   X=X+data[50:100, 1:6]
elif(class1=='Chinstrap'):
   X=X+data[100:150, 1:6]
if(class2=='Adelie'):
   X=np.append(X,data[0:50,1:6])
elif(class2=='Gentoo'):
   X=np.append(X, data[50:100, 1:6])
elif(class2=='Chinstrap'):
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

