# %% [markdown]
# 
# AI&DA18 Lab Exam
# Name: Atharva M. Kulkarni
# GR No: 11810384
# 
# Q1: Read the Dataset file “train_P1.csv” from the dataset folder. Design and train machine learning model on
# given columns(X-independent, y-Dependent). Perform following task in jupyter
# 
# a. Fill the missing values
# 
# b. Decide whether to do Regression or classification. Use at least 2 algorithms for machine learning
# model
# 
# c. Split data in train test split (80:20 proportion)
# 
# d. Find out the train score, test score of selected models and compare them.
# 

# %%
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
from google.colab import drive
drive.mount('/content/drive')

# %%
df = pd.read_csv('/content/drive/My Drive/Datasets/train_P1.csv')

# %% [markdown]
# 
# Filling Nan with mean X values.
# (Although median values would have been a better metric on paper, the model with mean values seems to perform better)
# 

# %%
print(df.head(10))
print(len(df['x']))
print(len(df['y']))
display(df.iloc[155])

X = df.drop(['y'], 1)
y = df['y']
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)
display(X.iloc[155])

# %%
X = np.array(X)
y = np.array(y)

# %%
X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, test_size = 0.2, shuffle = True)

# %% [markdown]
# 
# This is clearly a Regression Problem given the nature of the dependent variable
# The two regression methods implemented here are:
# 
# Random Forest Regressor
# 
# Support Vector Regressor
# 

# %%
clf = RandomForestRegressor(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
print("Train Accuracy: ", clf.score(X_train, y_train))
print("Test Accuracy: ", clf.score(X_test, y_test))

# %%
clf = svm.SVR()
clf.fit(X_train, y_train)
print("Train Accuracy: ", clf.score(X_train, y_train))
print("Test Accuracy: ", clf.score(X_test, y_test))


