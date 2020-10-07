#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from itertools import combinations


# In[10]:


data = pd.read_csv(r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\CCPP\Fold1.csv')
temp = np.split(data[:-3], 5)
data1 = temp[0]
data2 = temp[1]
data3 = temp[2]
data4 = temp[3]
data5 = temp[4]


# In[11]:


comb_error = []
X = []
Y = []
X.append(np.vstack((data1.values[:,0],data1.values[:,1],data1.values[:,2],data1.values[:,3],np.ones(data1.values[:,0].shape[0]))).T)
X.append(np.vstack((data2.values[:,0],data2.values[:,1],data2.values[:,2],data2.values[:,3],np.ones(data2.values[:,0].shape[0]))).T)
X.append(np.vstack((data3.values[:,0],data3.values[:,1],data3.values[:,2],data3.values[:,3],np.ones(data3.values[:,0].shape[0]))).T)
X.append(np.vstack((data4.values[:,0],data4.values[:,1],data4.values[:,2],data4.values[:,3],np.ones(data4.values[:,0].shape[0]))).T)
X.append(np.vstack((data5.values[:,0],data5.values[:,1],data5.values[:,2],data5.values[:,3],np.ones(data5.values[:,0].shape[0]))).T)
Y.append(data1.values[:,4].T)
Y.append(data2.values[:,4].T)
Y.append(data3.values[:,4].T)
Y.append(data4.values[:,4].T)
Y.append(data5.values[:,4].T)

error = []
poli_error = []
for i in range (5):
        cfv = [0,1,2,3,4]
        del cfv[4-i]
        X1 = np.vstack((X[cfv[0]],X[cfv[1]],X[cfv[2]],X[cfv[3]]))
        Y1 = np.hstack((Y[cfv[0]],Y[cfv[1]],Y[cfv[2]],Y[cfv[3]]))

        YY1 = np.matrix(Y1,dtype = 'float').T

        W1 = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(YY1) #W = (X'X)^-1(X'Y)

        X_range = np.linspace(X1.argmax(),X1.argmin(),2)
        line = np.array(W1[0] + W1[1] * X_range)

        #    plt.figure(i)
        #    plt.plot(X_range, line.T, color = 'r')
        #    plt.scatter(X1[:,1], Y1)

        error.append(np.sum(np.square(Y[i] - W1.T * X[i].T)))

poli_error.append(mean(error))
print ("all features combined error" ,mean(error))
comb_error.append(poli_error)


# In[12]:


X = []
Y = []
X.append(np.vstack((data1.values[:,0],data1.values[:,1],data1.values[:,2],data1.values[:,3],np.square((data1.values[:,0],data1.values[:,1],data1.values[:,2],data1.values[:,3])),np.ones(data1.values[:,0].shape[0]))).T)
X.append(np.vstack((data2.values[:,0],data2.values[:,1],data2.values[:,2],data2.values[:,3],np.square((data2.values[:,0],data2.values[:,1],data2.values[:,2],data2.values[:,3])),np.ones(data2.values[:,0].shape[0]))).T)
X.append(np.vstack((data3.values[:,0],data3.values[:,1],data3.values[:,2],data3.values[:,3],np.square((data3.values[:,0],data3.values[:,1],data3.values[:,2],data3.values[:,3])),np.ones(data3.values[:,0].shape[0]))).T)
X.append(np.vstack((data4.values[:,0],data4.values[:,1],data4.values[:,2],data4.values[:,3],np.square((data4.values[:,0],data4.values[:,1],data4.values[:,2],data4.values[:,3])),np.ones(data4.values[:,0].shape[0]))).T)
X.append(np.vstack((data5.values[:,0],data5.values[:,1],data5.values[:,2],data5.values[:,3],np.square((data5.values[:,0],data5.values[:,1],data5.values[:,2],data5.values[:,3])),np.ones(data5.values[:,0].shape[0]))).T)
Y.append(data1.values[:,4].T)
Y.append(data2.values[:,4].T)
Y.append(data3.values[:,4].T)
Y.append(data4.values[:,4].T)
Y.append(data5.values[:,4].T)

error = []
poli_error = []
for i in range (5):
        cfv = [0,1,2,3,4]
        del cfv[4-i]
        X1 = np.vstack((X[cfv[0]],X[cfv[1]],X[cfv[2]],X[cfv[3]]))
        Y1 = np.hstack((Y[cfv[0]],Y[cfv[1]],Y[cfv[2]],Y[cfv[3]]))

        YY1 = np.matrix(Y1,dtype = 'float').T

        W1 = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(YY1) #W = (X'X)^-1(X'Y)

        X_range = np.linspace(X1.argmax(),X1.argmin(),2)
        line = np.array(W1[0] + W1[1] * X_range)

        #    plt.figure(i)
        #    plt.plot(X_range, line.T, color = 'r')
        #    plt.scatter(X1[:,1], Y1)

        error.append(np.sum(np.square(Y[i] - W1.T * X[i].T)))

poli_error.append(mean(error))
print ("all features combined error" ,mean(error))
comb_error.append(poli_error)


# In[13]:


X = []
Y = []
X.append(np.vstack((data1.values[:,0],data1.values[:,1],data1.values[:,2],data1.values[:,3],np.square((data1.values[:,0],data1.values[:,1],data1.values[:,2],data1.values[:,3])),np.power((data1.values[:,0],data1.values[:,1],data1.values[:,2],data1.values[:,3]),3),np.ones(data1.values[:,0].shape[0]))).T)
X.append(np.vstack((data2.values[:,0],data2.values[:,1],data2.values[:,2],data2.values[:,3],np.square((data2.values[:,0],data2.values[:,1],data2.values[:,2],data2.values[:,3])),np.power((data2.values[:,0],data2.values[:,1],data2.values[:,2],data2.values[:,3]),3),np.ones(data2.values[:,0].shape[0]))).T)
X.append(np.vstack((data3.values[:,0],data3.values[:,1],data3.values[:,2],data3.values[:,3],np.square((data3.values[:,0],data3.values[:,1],data3.values[:,2],data3.values[:,3])),np.power((data3.values[:,0],data3.values[:,1],data3.values[:,2],data3.values[:,3]),3),np.ones(data3.values[:,0].shape[0]))).T)
X.append(np.vstack((data4.values[:,0],data4.values[:,1],data4.values[:,2],data4.values[:,3],np.square((data4.values[:,0],data4.values[:,1],data4.values[:,2],data4.values[:,3])),np.power((data4.values[:,0],data4.values[:,1],data4.values[:,2],data4.values[:,3]),3),np.ones(data4.values[:,0].shape[0]))).T)
X.append(np.vstack((data5.values[:,0],data5.values[:,1],data5.values[:,2],data5.values[:,3],np.square((data5.values[:,0],data5.values[:,1],data5.values[:,2],data5.values[:,3])),np.power((data5.values[:,0],data5.values[:,1],data5.values[:,2],data5.values[:,3]),3),np.ones(data5.values[:,0].shape[0]))).T)
Y.append(data1.values[:,4].T)
Y.append(data2.values[:,4].T)
Y.append(data3.values[:,4].T)
Y.append(data4.values[:,4].T)
Y.append(data5.values[:,4].T)

error = []
poli_error = []
for i in range (5):
        cfv = [0,1,2,3,4]
        del cfv[4-i]
        X1 = np.vstack((X[cfv[0]],X[cfv[1]],X[cfv[2]],X[cfv[3]]))
        Y1 = np.hstack((Y[cfv[0]],Y[cfv[1]],Y[cfv[2]],Y[cfv[3]]))

        YY1 = np.matrix(Y1,dtype = 'float').T

        W1 = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(YY1) #W = (X'X)^-1(X'Y)

        X_range = np.linspace(X1.argmax(),X1.argmin(),2)
        line = np.array(W1[0] + W1[1] * X_range)

        #    plt.figure(i)
        #    plt.plot(X_range, line.T, color = 'r')
        #    plt.scatter(X1[:,1], Y1)

        error.append(np.sum(np.square(Y[i] - W1.T * X[i].T)))

poli_error.append(mean(error))
print ("all features combined error" ,mean(error))
comb_error.append(poli_error)


# In[14]:


X = []
Y = []
X.append(np.vstack((data1.values[:,0],data1.values[:,1],data1.values[:,2],data1.values[:,3],np.square((data1.values[:,0],data1.values[:,1],data1.values[:,2],data1.values[:,3])),np.power((data1.values[:,0],data1.values[:,1],data1.values[:,2],data1.values[:,3]),3),np.power((data1.values[:,0],data1.values[:,1],data1.values[:,2],data1.values[:,3]),4),np.ones(data1.values[:,0].shape[0]))).T)
X.append(np.vstack((data2.values[:,0],data2.values[:,1],data2.values[:,2],data2.values[:,3],np.square((data2.values[:,0],data2.values[:,1],data2.values[:,2],data2.values[:,3])),np.power((data2.values[:,0],data2.values[:,1],data2.values[:,2],data2.values[:,3]),3),np.power((data2.values[:,0],data2.values[:,1],data2.values[:,2],data2.values[:,3]),4),np.ones(data2.values[:,0].shape[0]))).T)
X.append(np.vstack((data3.values[:,0],data3.values[:,1],data3.values[:,2],data3.values[:,3],np.square((data3.values[:,0],data3.values[:,1],data3.values[:,2],data3.values[:,3])),np.power((data3.values[:,0],data3.values[:,1],data3.values[:,2],data3.values[:,3]),3),np.power((data3.values[:,0],data3.values[:,1],data3.values[:,2],data3.values[:,3]),4),np.ones(data3.values[:,0].shape[0]))).T)
X.append(np.vstack((data4.values[:,0],data4.values[:,1],data4.values[:,2],data4.values[:,3],np.square((data4.values[:,0],data4.values[:,1],data4.values[:,2],data4.values[:,3])),np.power((data4.values[:,0],data4.values[:,1],data4.values[:,2],data4.values[:,3]),3),np.power((data4.values[:,0],data4.values[:,1],data4.values[:,2],data4.values[:,3]),4),np.ones(data4.values[:,0].shape[0]))).T)
X.append(np.vstack((data5.values[:,0],data5.values[:,1],data5.values[:,2],data5.values[:,3],np.square((data5.values[:,0],data5.values[:,1],data5.values[:,2],data5.values[:,3])),np.power((data5.values[:,0],data5.values[:,1],data5.values[:,2],data5.values[:,3]),3),np.power((data5.values[:,0],data5.values[:,1],data5.values[:,2],data5.values[:,3]),4),np.ones(data5.values[:,0].shape[0]))).T)
Y.append(data1.values[:,4].T)
Y.append(data2.values[:,4].T)
Y.append(data3.values[:,4].T)
Y.append(data4.values[:,4].T)
Y.append(data5.values[:,4].T)

error = []
poli_error = []
for i in range (5):
        cfv = [0,1,2,3,4]
        del cfv[4-i]
        X1 = np.vstack((X[cfv[0]],X[cfv[1]],X[cfv[2]],X[cfv[3]]))
        Y1 = np.hstack((Y[cfv[0]],Y[cfv[1]],Y[cfv[2]],Y[cfv[3]]))

        YY1 = np.matrix(Y1,dtype = 'float').T

        W1 = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(YY1) #W = (X'X)^-1(X'Y)

        X_range = np.linspace(X1.argmax(),X1.argmin(),2)
        line = np.array(W1[0] + W1[1] * X_range)

        #    plt.figure(i)
        #    plt.plot(X_range, line.T, color = 'r')
        #    plt.scatter(X1[:,1], Y1)

        error.append(np.sum(np.square(Y[i] - W1.T * X[i].T)))

poli_error.append(mean(error))
print ("all features combined error" ,mean(error))
comb_error.append(poli_error)


# In[15]:


comb_error


# In[16]:


xaxis = ["1st deg","2nd deg","3rd deg"]
plt.plot(xaxis,comb_error[0:3])


# In[ ]:




