#!/usr/bin/env python
# coding: utf-8

# # create models and compare their score OR estimate their accuracy on unseen data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

get_ipython().run_line_magic('matplotlib', 'inline')
import os

print(xgboost.__version__)
os.getcwd()
os.listdir(os.getcwd())


# In[2]:


#read file from local :

missing_value_formats=["n.a.","?","NA","n/a", "na", "--"]
df=pd.read_csv('Lending_Club_Loan_approval_Optimization.csv')
print(df)


# In[3]:


df.head


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[3]:


#create a validation set 
#We will split the loaded dataset into two, 80% of which we will use to train,
#evaluate and select among our models, and 20% that we will hold back as a validation dataset.

#we are going to hold back that 20% data that the algorithms will not get to see and we will use this 
#data to get a second and independent idea of how accurate the best model might actually be.

# Split-out validation dataset
array = df.values
print(array)
X = array[:,0:5]
y = array[:,5]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# In[4]:


#Test Harness
#We will use stratified 10-fold cross validation to estimate model accuracy.

#This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for 
#all combinations of train-test splits.

#Stratified means that each fold or split of the dataset will aim to have the same distribution of 
#example by class as exist in the whole training dataset.

#We are using the metric of ‘accuracy‘ to evaluate models.

#This is a ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate).
#We will be using the scoring variable when we run build and evaluate each model next.

#We don’t know which algorithms would be good on this problem or what configurations to use.
#We get an idea from the plots.

#Let’s test 6 different algorithms:

#Linear - [Logistic Regression (LR),Polynomial,Stepwise,Ridge,Lasso,ElasticNet].
#Linear Discriminant Analysis (LDA)
#K-Nearest Neighbors (KNN).
#Classification and Regression Trees (CART).
#Gaussian Naive Bayes (NB).
#Support Vector Machines (SVM).
#XGBoost 

#This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.
#Let’s build and evaluate our models:

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('xgBoost',XGBClassifier(use_label_encoder=False)))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[ ]:


#Make Predictions
#We must choose an algorithm to use to make predictions.

#The results in the previous section suggest that the xgboost was perhaps the most accurate model. We will use this model as our final model.

#Now we want to get an idea of the accuracy of the model on our validation set.

#This will give us an independent final check on the accuracy of the best model. It is valuable to keep a validation set just in case you made a slip during training, such as overfitting to the training set or a data leak. Both of these issues will result in an overly optimistic result.

#6.1 Make Predictions
#We can fit the model on the entire training dataset and make predictions on the validation dataset OR test dataset.

# Make predictions on validation dataset
model = XGBClassifier(use_label_encoder=False)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
#You might also like to make predictions for single rows of data. For examples on how to do that, see the tutorial:

#How to Make Predictions with scikit-learn
#You might also like to save the model to file and load it later to make predictions on new data. For examples on how to do this, see the tutorial:


# In[8]:



# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:




