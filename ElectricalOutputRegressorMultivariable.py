#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from itertools import combinations


# In[2]:


data = pd.read_csv(r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\CCPP\Fold1.csv')
temp = np.split(data[:-3], 5)
data1 = temp[0]
data2 = temp[1]
data3 = temp[2]
data4 = temp[3]
data5 = temp[4]


# In[3]:


comb_error = []
for ii in range (4):
    for jj in range ((ii+1),4):
        X = []
        Y = []
        X.append(np.vstack((data1.values[:,ii],data1.values[:,jj],np.ones(data1.values[:,0].shape[0]))).T)
        X.append(np.vstack((data2.values[:,ii],data2.values[:,jj],np.ones(data1.values[:,0].shape[0]))).T)
        X.append(np.vstack((data3.values[:,ii],data3.values[:,jj],np.ones(data1.values[:,0].shape[0]))).T)
        X.append(np.vstack((data4.values[:,ii],data4.values[:,jj],np.ones(data1.values[:,0].shape[0]))).T)
        X.append(np.vstack((data5.values[:,ii],data5.values[:,jj],np.ones(data1.values[:,0].shape[0]))).T)
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
        print ("feature",ii, "and",jj,"error: " ,mean(error))
        comb_error.append(poli_error)


# In[4]:


lst = [0,1,2,3]
for combo in combinations(lst, 3):
    X = []
    Y = []
    X.append(np.vstack((data1.values[:,combo[0]],data1.values[:,combo[1]],data1.values[:,combo[2]],np.ones(data1.values[:,0].shape[0]))).T)
    X.append(np.vstack((data2.values[:,combo[0]],data2.values[:,combo[1]],data2.values[:,combo[2]],np.ones(data2.values[:,0].shape[0]))).T)
    X.append(np.vstack((data3.values[:,combo[0]],data3.values[:,combo[1]],data3.values[:,combo[2]],np.ones(data3.values[:,0].shape[0]))).T)
    X.append(np.vstack((data4.values[:,combo[0]],data4.values[:,combo[1]],data4.values[:,combo[2]],np.ones(data4.values[:,0].shape[0]))).T)
    X.append(np.vstack((data5.values[:,combo[0]],data5.values[:,combo[1]],data5.values[:,combo[2]],np.ones(data5.values[:,0].shape[0]))).T)
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


# In[ ]:





# In[5]:



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


# In[6]:


comb_error
xaxis = ["AT&V","AT&AP","AT&RH","V&AP","V&RH", "AP&RH","AT,V,AP","AT,V,RH","AT,AP,RH","V,AP,RH","ALL"]
plt.rcParams["figure.figsize"] = (20,10)
plt.plot(xaxis,comb_error) 


# In[7]:


comb_error


# In[ ]:




