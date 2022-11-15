#!/usr/bin/env python
# coding: utf-8

# In[22]:


#importing required libraries
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sbn 


# In[8]:


#loading the dataset
dataset= pd.read_csv("C:/Users/user/OneDrive/Desktop/A.I/assignment 2/exams.csv")


# In[3]:


#exploratory data analysis
dataset.shape


# In[9]:



dataset.head(5)


# In[10]:


dataset.info()


# In[12]:


dataset.value_counts()


# In[16]:


plt.scatter(dataset['math score'],dataset['reading score'], dataset['writing score'])
plt.show()


# In[18]:


scores=['math score','reading score','writing score']
x = dataset[scores].values
y = dataset['test preparation course'].values


# In[43]:


from sklearn import tree
#import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# In[36]:


clf = tree.DecisionTreeClassifier()


# In[32]:


#data splitting
Xtrain, Xtest, Ytrain,Ytest= train_test_split(x,y,test_size=0.4, random_state=75)


# In[47]:


Xt,Xcv,Yt,Ycv = train_test_split(Xtrain,Ytrain, test_size=0.10, random_state=75)


# In[39]:


#creating the decision tree
dataset_clf=DecisionTreeClassifier()
dataset_clf.fit(Xt,Yt)
#visualizéthe tree
tree.plot_tree(dataset_clf)


# In[40]:


#accuracy
print('Accuracy score is: ',cross_val_score(dataset_clf,Xt,Yt, cv=3, scoring='accuracy').mean())


# In[44]:


#checking validation test data on our trained model and getting performance metrices
Y_hat= dataset_clf.predict(Xcv)


# In[48]:


print('Accuracy score for validation test data is: ',accuracy_score(Ycv,Y_hat))
multilabel_confusion_matrix(Ycv,Y_hat)


# In[49]:


Yt_hat =dataset_clf.predict(Xtest)
Yt_hat


# In[51]:


print('Model accuracy Score on totally unseen data(Xtest) is',accuracy_score(Ytest,Yt_hat)*100,'%')
multilabel_confusion_matrix(Ytest, Yt_hat)


# In[54]:


dataset_fclf=DecisionTreeClassifier()
dataset_fclf.fit(Xtrain,Ytrain)
tree.plot_tree(dataset_fclf)


# In[55]:


Yt_flat= dataset_fclf.predict(Xtest)
Yt_flat
print('Model accuracy score on totally unseen data(Xtest) is:',accuracy_score(Ytest,Yt_flat)*100,'%')
multilabel_confusion_matrix(Ytest,Yt_flat)


# In[ ]:




