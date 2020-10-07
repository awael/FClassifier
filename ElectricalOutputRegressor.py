#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean 


# In[16]:


data = pd.read_csv(r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\CCPP\Fold1.csv')


# In[17]:


temp = np.split(data[:-3], 5)


# In[18]:


data1 = temp[0]
data2 = temp[1]
data3 = temp[2]
data4 = temp[3]
data5 = temp[4]


# In[19]:


X = []
Y = []
X.append(data1.values[:,1])
X.append(data2.values[:,1])
X.append(data3.values[:,1])
X.append(data4.values[:,1])
X.append(data5.values[:,1])
Y.append(data1.values[:,4])
Y.append(data2.values[:,4])
Y.append(data3.values[:,4])
Y.append(data4.values[:,4])
Y.append(data5.values[:,4])


# In[20]:


error = []
poli_error = []
for i in range (5):
    cfv = [0,1,2,3,4]
    del cfv[4-i]
    X1 = np.concatenate((X[cfv[0]],X[cfv[1]],X[cfv[2]],X[cfv[3]]),axis = 0)
    Y1 = np.concatenate((Y[cfv[0]],Y[cfv[1]],Y[cfv[2]],Y[cfv[3]]),axis = 0)
    print ("folds: ",cfv)
    XX1 = np.matrix([np.ones(np.shape(X1)[0]), X1]).T
    YY1 = np.matrix(Y1).T

    W1 = np.linalg.inv(XX1.T.dot(XX1)).dot(XX1.T).dot(YY1) #W = (X'X)^-1(X'Y)

    X_range = np.linspace(max(X1),min(X1),2)
    line = np.array(W1[0] + W1[1] * X_range)

    plt.figure(i)
    plt.plot(X_range, line.T, color = 'r')
    plt.scatter(X1, Y1)
    
    error.append(np.sum(np.square((W1[0] + W1[1] * X[i]) - Y[i])))

    plt.show()
poli_error.append(mean(error))
print ("1st degree polinomial average error = ",mean(error))


# In[ ]:





# In[21]:


#Second degree
error=[]
for i in range (5):
    cfv = [0,1,2,3,4]
    del cfv[4-i]

    X1 = np.concatenate((X[cfv[0]],X[cfv[1]],X[cfv[2]],X[cfv[3]]),axis = 0)
    Y1 = np.concatenate((Y[cfv[0]],Y[cfv[1]],Y[cfv[2]],Y[cfv[3]]),axis = 0)
    print ("folds: ",cfv)

    XX1 = np.matrix([np.ones(np.shape(X1)[0]), X1,np.square(X1)]).T
    YY1 = np.matrix(Y1).T

    W1 = np.linalg.inv(XX1.T.dot(XX1)).dot(XX1.T).dot(YY1) #W = (X'X)^-1(X'Y)

    X_range = np.linspace(max(X1),min(X1),1000)
    line = np.array(W1[0]*np.power(X_range,0) + W1[1]*np.power(X_range,1) + W1[2]*np.power(X_range,2))

    plt.figure(i)
    plt.plot(X_range, line.T, color = 'r')
    plt.scatter(X1, Y1)
    
    error.append(np.sum(np.square((W1[0] + W1[1] * X[i] + W1[2]*np.power(X[i],2)) - Y[i])))
    
    plt.show()
poli_error.append(mean(error))
print ("2nd degree polinomial average error = ",mean(error))
print (error)


# In[22]:


#third degree
error=[]
for i in range (5):
    cfv = [0,1,2,3,4]
    del cfv[4-i]

    X1 = np.concatenate((X[cfv[0]],X[cfv[1]],X[cfv[2]],X[cfv[3]]),axis = 0)
    Y1 = np.concatenate((Y[cfv[0]],Y[cfv[1]],Y[cfv[2]],Y[cfv[3]]),axis = 0)
    print ("folds: ",cfv)

    XX1 = np.matrix([np.ones(np.shape(X1)[0]), X1,np.square(X1),np.power(X1,3)]).T
    YY1 = np.matrix(Y1).T

    W1 = np.linalg.inv(XX1.T.dot(XX1)).dot(XX1.T).dot(YY1) #W = (X'X)^-1(X'Y)

    X_range = np.linspace(max(X1),min(X1),1000)
    line = np.array(W1[0]*np.power(X_range,0) + W1[1]*np.power(X_range,1) + W1[2]*np.power(X_range,2) + W1[3]*np.power(X_range,3))

    plt.figure(i)
    plt.plot(X_range, line.T, color = 'r')
    plt.scatter(X1, Y1)
    
    error.append(np.sum(np.square((W1[0] + W1[1] * X[i] + W1[2]*np.power(X[i],2) + W1[3]*np.power(X[i],3)) - Y[i])))
    
    plt.show()
poli_error.append(mean(error))
print ("3rd degree polinomial average error = ",mean(error))
print (error)


# In[23]:


#fourth degree
error=[]
for i in range (5):
    cfv = [0,1,2,3,4]
    del cfv[4-i]

    X1 = np.concatenate((X[cfv[0]],X[cfv[1]],X[cfv[2]],X[cfv[3]]),axis = 0)
    Y1 = np.concatenate((Y[cfv[0]],Y[cfv[1]],Y[cfv[2]],Y[cfv[3]]),axis = 0)
    print ("folds: ",cfv)

    XX1 = np.matrix([np.ones(np.shape(X1)[0]), X1,np.square(X1),np.power(X1,3),np.power(X1,4)]).T
    YY1 = np.matrix(Y1).T

    W1 = np.linalg.inv(XX1.T.dot(XX1)).dot(XX1.T).dot(YY1) #W = (X'X)^-1(X'Y)

    X_range = np.linspace(max(X1),min(X1),1000)
    line = np.array(W1[0]*np.power(X_range,0) + W1[1]*np.power(X_range,1) + W1[2]*np.power(X_range,2) + W1[3]*np.power(X_range,3) + W1[4]*np.power(X_range,4))

    plt.figure(i)
    plt.plot(X_range, line.T, color = 'r')
    plt.scatter(X1, Y1)
    
    error.append(np.sum(np.square((W1[0] + W1[1] * X[i] + W1[2]*np.power(X[i],2) + W1[3]*np.power(X[i],3) + W1[4]*np.power(X[i],4)) - Y[i])))
    
    plt.show()
poli_error.append(mean(error))
print ("4th degree polinomial average error = ",mean(error))
print (error)


# In[24]:


#fifth degree
error=[]
for i in range (5):
    cfv = [0,1,2,3,4]
    del cfv[4-i]

    X1 = np.concatenate((X[cfv[0]],X[cfv[1]],X[cfv[2]],X[cfv[3]]),axis = 0)
    Y1 = np.concatenate((Y[cfv[0]],Y[cfv[1]],Y[cfv[2]],Y[cfv[3]]),axis = 0)
    print ("folds: ",cfv)

    XX1 = np.matrix([np.ones(np.shape(X1)[0]), X1,np.square(X1),np.power(X1,3),np.power(X1,4),np.power(X1,5)]).T
    YY1 = np.matrix(Y1).T

    W1 = np.linalg.inv(XX1.T.dot(XX1)).dot(XX1.T).dot(YY1) #W = (X'X)^-1(X'Y)

    X_range = np.linspace(max(X1),min(X1),1000)
    line = np.array(W1[0]*np.power(X_range,0) + W1[1]*np.power(X_range,1) + W1[2]*np.power(X_range,2) + W1[3]*np.power(X_range,3) + W1[4]*np.power(X_range,4) + W1[5]*np.power(X_range,5))

    plt.figure(i)
    plt.plot(X_range, line.T, color = 'r')
    plt.scatter(X1, Y1)
    
    error.append(np.sum(np.square((W1[0] + W1[1] * X[i] + W1[2]*np.power(X[i],2) + W1[3]*np.power(X[i],3) + W1[4]*np.power(X[i],4) + W1[5]*np.power(X[i],5)) - Y[i])))
    
    plt.show()
poli_error.append(mean(error))
print ("5th degree polinomial average error = ",mean(error))
print (error)


# In[25]:


#sixth degree
error=[]
for i in range (5):
    cfv = [0,1,2,3,4]
    del cfv[4-i]

    X1 = np.concatenate((X[cfv[0]],X[cfv[1]],X[cfv[2]],X[cfv[3]]),axis = 0)
    Y1 = np.concatenate((Y[cfv[0]],Y[cfv[1]],Y[cfv[2]],Y[cfv[3]]),axis = 0)
    print ("folds: ",cfv)

    XX1 = np.matrix([np.ones(np.shape(X1)[0]), X1,np.square(X1),np.power(X1,3),np.power(X1,4),np.power(X1,5),np.power(X1,6)]).T
    YY1 = np.matrix(Y1).T

    W1 = np.linalg.inv(XX1.T.dot(XX1)).dot(XX1.T).dot(YY1) #W = (X'X)^-1(X'Y)

    X_range = np.linspace(max(X1),min(X1),1000)
    line = np.array(W1[0]*np.power(X_range,0) + W1[1]*np.power(X_range,1) + W1[2]*np.power(X_range,2) + W1[3]*np.power(X_range,3) + W1[4]*np.power(X_range,4) + W1[5]*np.power(X_range,5) + W1[6]*np.power(X_range,6))

    plt.figure(i)
    plt.plot(X_range, line.T, color = 'r')
    plt.scatter(X1, Y1)
    
    error.append(np.sum(np.square((W1[0] + W1[1] * X[i] + W1[2]*np.power(X[i],2) + W1[3]*np.power(X[i],3) + W1[4]*np.power(X[i],4) + W1[5]*np.power(X[i],5) + W1[6]*np.power(X[i],6)) - Y[i])))
    
    plt.show()
poli_error.append(mean(error))
print ("6th degree polinomial average error = ",mean(error))
print (error)


# In[26]:


#seventh degree
error=[]
for i in range (5):
    cfv = [0,1,2,3,4]
    del cfv[4-i]

    X1 = np.concatenate((X[cfv[0]],X[cfv[1]],X[cfv[2]],X[cfv[3]]),axis = 0)
    Y1 = np.concatenate((Y[cfv[0]],Y[cfv[1]],Y[cfv[2]],Y[cfv[3]]),axis = 0)
    print ("folds: ",cfv)

    XX1 = np.matrix([np.ones(np.shape(X1)[0]), X1,np.square(X1),np.power(X1,3),np.power(X1,4),np.power(X1,5),np.power(X1,6),np.power(X1,6)]).T
    YY1 = np.matrix(Y1).T

    W1 = np.linalg.inv(XX1.T.dot(XX1)).dot(XX1.T).dot(YY1) #W = (X'X)^-1(X'Y)

    X_range = np.linspace(max(X1),min(X1),1000)
    line = np.array(W1[0]*np.power(X_range,0) + W1[1]*np.power(X_range,1) + W1[2]*np.power(X_range,2) + W1[3]*np.power(X_range,3) + W1[4]*np.power(X_range,4) + W1[5]*np.power(X_range,5) + W1[6]*np.power(X_range,6) + W1[7]*np.power(X_range,7))

    plt.figure(i)
    plt.plot(X_range, line.T, color = 'r')
    plt.scatter(X1, Y1)
    
    error.append(np.sum(np.square((W1[0] + W1[1] * X[i] + W1[2]*np.power(X[i],2) + W1[3]*np.power(X[i],3) + W1[4]*np.power(X[i],4) + W1[5]*np.power(X[i],5) + W1[6]*np.power(X[i],6) + W1[7]*np.power(X[i],7)) - Y[i])))
    
    plt.show()
poli_error.append(mean(error))
print ("7th degree polinomial average error = ",mean(error))
print (error)


# In[37]:


plt.plot(np.linspace(1,6,6),poli_error[0:6]) #error least at 6th order polinomial


# In[ ]:





# In[ ]:




