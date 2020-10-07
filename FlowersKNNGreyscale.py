#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
from tqdm import tqdm
import statistics 
from statistics import mode
from collections import Counter 
from statistics import mean 


# In[2]:


FLOWER_DAISY_DIR = r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\flower_photos\daisy'
FLOWER_SUNFLOWER_DIR =r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\flower_photos\sunflowers'
FLOWER_TULIP_DIR = r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\flower_photos\tulips'
FLOWER_DANDI_DIR = r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\flower_photos\dandelion'
FLOWER_ROSE_DIR = r'C:\Users\ahmed\OneDrive\Documents\FALL 20\Deep Learning\flower_photos\roses'


# In[3]:


sorted =os.listdir(FLOWER_DAISY_DIR)
sorted.sort()
len(sorted[:-100])


# In[20]:


X=[]
Y=[]
X_test=[]
Y_test=[]
def assign_label(img,flower_type):
    return flower_type
def make_train_test_data(flower_type,DIR):
  sorted =os.listdir(DIR)
  sorted.sort()
  for train_img in tqdm(sorted[-100:]):
    label=assign_label(train_img,flower_type)
    path = os.path.join(DIR,train_img)
    ti = cv2.imread(path,cv2.IMREAD_GRAYSCALE) #IMREAD_GRAYSCALE for grayscale
    ti = cv2.resize(ti, (40,40))
    
    X_test.append(np.array(ti))
    Y_test.append(str(label))

  len(sorted[:-100]) #all but last 100 images in file
  for img in tqdm(sorted[:-100]):
    label=assign_label(img,flower_type)
    path = os.path.join(DIR,img)
    i = cv2.imread(path,cv2.IMREAD_GRAYSCALE) #IMREAD_GRAYSCALE for grayscale
    i = cv2.resize(i, (40,40))
        
    X.append(np.array(i))
    Y.append(str(label))


# In[21]:


make_train_test_data('Daisy',FLOWER_DAISY_DIR)
print(len(X))


# In[22]:


make_train_test_data('Rose',FLOWER_ROSE_DIR)
print(len(X))


# In[23]:


make_train_test_data('Sunflower',FLOWER_SUNFLOWER_DIR)
print(len(X))


# In[24]:


make_train_test_data('Tulip',FLOWER_TULIP_DIR)
print(len(X))


# In[25]:


make_train_test_data('Dandelion',FLOWER_DANDI_DIR)
print(len(X))


# In[27]:


XX=[]
XX_test=[]
for image in X:
    XX.append(image.reshape(40*40))
for image in X_test:
    XX_test.append(image.reshape(40*40))
XX_np = np.array(XX)
XX_test_np = np.array(XX_test)
Y_test_np = (np.array(Y_test))
Y_np = np.array(Y)

from sklearn.utils import shuffle
X_shuffled, Y_shuffled = shuffle(XX_np, Y_np, random_state=0)


# In[28]:


X_shuffled_test, Y_shuffled_test = shuffle(XX_test_np, Y_test_np, random_state=0)


# In[31]:


print(X_shuffled[1,:].shape)
print (X_shuffled.shape, Y_shuffled.shape)


# In[32]:


def knn(X_data,X_input,n,labels):
    num_test = X_input.shape[0]
    distances=[]
    NNearestIndex = []
    rows, cols = (num_test, n) 
    NNearestName = [[0 for i in range(cols)] for j in range(rows)]
    result = []
    
    for i in range(num_test):
        distances.append(np.sqrt(np.sum(np.square(X_data - X_input[i,:]), axis = 1)))
        NNearestIndex.append(distances[i].argsort()[:n])
        for j in range(n):
            NNearestName[i][j]=labels[NNearestIndex[i][j]]
        occurence_count = Counter(NNearestName[i]) 
        result.append(occurence_count.most_common(1)[0][0]) 
       
    return result


# In[33]:


pred_result_avg=[]
X_folds = np.array_split(X_shuffled,5)
Y_folds = np.array_split(Y_shuffled,5)
rows, cols = (5, 15) 
pred_result = [[0 for i in range(cols)] for j in range(rows)]
for k in tqdm(range(15)):
    pred_result[0][k] = knn(np.concatenate((X_folds[0],X_folds[1],X_folds[2],X_folds[3])),X_folds[4],k+1,np.concatenate((Y_folds[0],Y_folds[1],Y_folds[2],Y_folds[3])).tolist())
    pred_result[1][k] = knn(np.concatenate((X_folds[0],X_folds[1],X_folds[2],X_folds[4])),X_folds[3],k+1,np.concatenate((Y_folds[0],Y_folds[1],Y_folds[2],Y_folds[4])).tolist())
    pred_result[2][k] = knn(np.concatenate((X_folds[0],X_folds[1],X_folds[3],X_folds[4])),X_folds[2],k+1,np.concatenate((Y_folds[0],Y_folds[1],Y_folds[3],Y_folds[4])).tolist())
    pred_result[3][k] = knn(np.concatenate((X_folds[0],X_folds[2],X_folds[3],X_folds[4])),X_folds[1],k+1,np.concatenate((Y_folds[0],Y_folds[2],Y_folds[3],Y_folds[4])).tolist())
    pred_result[4][k] = knn(np.concatenate((X_folds[1],X_folds[2],X_folds[3],X_folds[4])),X_folds[0],k+1,np.concatenate((Y_folds[1],Y_folds[2],Y_folds[3],Y_folds[4])).tolist())    


# In[34]:


cross_fold_accuracy = []
for k in range (15):
    fold_accuracy = []
    for i in range (5):
        positives = 0
        for j in range (len(X_folds[0])):
            if (Y_folds[4-i][j]==pred_result[i][k][j]):
                positives=positives+1
        fold_accuracy.append(positives/len(X_folds[0]))
    cross_fold_accuracy.append(mean(fold_accuracy)*100)    


# In[35]:


cross_fold_accuracy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(cross_fold_accuracy)
plt.ylabel('accuracy %')
plt.xlabel('K')
plt.show()


# In[36]:


cross_fold_accuracy


# In[37]:


final_result = []
final_result = knn(X_shuffled,X_shuffled_test,4,Y_shuffled)


# In[38]:


Daisy_positives = 0
Dandelion_positives = 0
Sunflower_positives = 0
Tulip_positives = 0
Rose_positives= 0
for i in range(500):
    if (final_result[i] == Y_shuffled_test[i]):
        if Y_shuffled_test[i] == 'Daisy':
            Daisy_positives+=1
        if Y_shuffled_test[i] == 'Dandelion':
            Dandelion_positives+=1
        if Y_shuffled_test[i] == 'Sunflower':
            Sunflower_positives+=1
        if Y_shuffled_test[i] == 'Tulip':
            Tulip_positives+=1
        if Y_shuffled_test[i] == 'Rose':
            Rose_positives+=1
Daisy_accuracy = (Daisy_positives/100) *100
Dandelion_accuracy = (Dandelion_positives/100) *100
Sunflower_accuracy = (Sunflower_positives/100) *100
Tulip_accuracy = (Tulip_positives/100) *100
Rose_accuracy = (Rose_positives/100) *100


# In[39]:


Daisy_accuracy


# In[40]:


print('accuracies:')
print('Daisy: ',Daisy_accuracy,'%,  Sunflower:', Sunflower_accuracy,'%,   Tulip: ', Tulip_accuracy, '%,   Dandelion: ', Dandelion_accuracy, '%,  Rose: ', Rose_accuracy,'%' )


# In[42]:


total_accuracy = (Daisy_accuracy+Dandelion_accuracy+Sunflower_accuracy+Tulip_accuracy+Rose_accuracy)/500 * 100
print('Average Correct Classification Rate: ', total_accuracy,'% - Best K = 4')
#Colored accuracy is higher as more features are retained, however data is only a third in greyscale,so prediction is faster

