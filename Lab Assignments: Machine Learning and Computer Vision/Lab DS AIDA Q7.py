# %% [markdown]
# 
# AI&DA18 Lab Exam
# Name: Atharva M. Kulkarni
# GR No: 11810384
# 
# Q7: Read the Dataset file “train_Titanic.csv” from the dataset folder. Design and train SVM machine learning
# model on given columns(Survived-Dependent Column). Perform following task in jupyter
# 
# a. Fill the missing values
# 
# b. Do Column encoding (If necessary)
# 
# c. Split data in train test split (70:30 proportion)
# 
# d. Create 3 models on Linear Kernel, Polynomial Kernel and RBF Kernel
# 
# e. Find out the train score, test score for all models and compare the results..
# 

# %%
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
import re
import os
from google.colab import drive
drive.mount('/content/drive')

# %%
df = pd.read_csv('/content/drive/My Drive/train.csv')

# %%
print(df.head(10))

# %% [markdown]
# 
# Analyzing all the different Features / Labels and displaying a count of Nan values in the dataset
# 

# %%
print(df.columns)
print(df.isnull().sum())

# %% [markdown]
# 
# The Cabin number could define where on the Ship was the person living and this in-turn could indicate whether the passenger had access to help immediately or whether he/she succumbed on impact.
# 
# So, here we categorize this column
# 

# %%
Cabin_set = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

df['Cabin'] = df['Cabin'].fillna("U0")
df['Cabin_set'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
df['Cabin_set'] = df['Cabin_set'].map(Cabin_set)
df['Cabin_set'] = df['Cabin_set'].fillna(0)

print(df.head())
print(df.columns)
print(df.isnull().sum())

# %% [markdown]
# 
# Replacing Nan 'Age' values with the median age
# 

# %%
df['Age'].fillna(df['Age'].median(), inplace=True)
print(df.isnull().sum())

# %% [markdown]
# 
# Dropping the redundant or less significant columns
# 

# %%
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], 1)
print(df.columns)

df['Fare'] = df['Fare'].astype(int)
df['Cabin_set'] = df['Cabin_set'].astype(int)
df['Age'] = df['Age'].astype(int)
display(df)

# %%
sex_numerical = {"male": 0, "female": 1}
df['Sex'] = df['Sex'].map(sex_numerical)
display(df)

# %%
X = df.drop(['Survived'], 1)
y = df['Survived']

# %%
X = np.array(X)
y = np.array(y)

# %%
X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, test_size = 0.3, shuffle = True)

# %% [markdown]
# 
# Here, we implemented SVM using the Linear, Polynomial and RBF kernels.
# 
# Results are summarized below
# 

# %%
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
print("Train Accuracy: ", clf.score(X_train, y_train))
print("Test Accuracy: ", clf.score(X_test, y_test))

# %%
clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)
print("Train Accuracy: ", clf.score(X_train, y_train))
print("Test Accuracy: ", clf.score(X_test, y_test))

# %%
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
print("Train Accuracy: ", clf.score(X_train, y_train))
print("Test Accuracy: ", clf.score(X_test, y_test))


