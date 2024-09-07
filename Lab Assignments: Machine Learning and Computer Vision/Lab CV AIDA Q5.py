# %% [markdown]
# **AI&DA18 Batch 1 Lab Exam Name: Atharva M. Kulkarni GR No: 11810384**
# 
# Q5: Write a Python Program to Implement Following
# Operations on File1.csv:
# 1. Standardization
# 2. Data Splitting (Training Data = 85%, Testing Data =15%, No Randomness
# 3. Use Support Vector Machine (SVM) Classifier to Classify the Data.
# Compute the Accuracy by Using Confusion Matrix
# 4. Use K-Nearest Neighbour (KNN) Classifier to Classify the Data. Compute
# the Accuracy by Using Confusion Matrix
# 5. Comment on Suitability of Supervised and Unsupervised Machine
# Learning Classifiers

# %%
import pandas as pd
import numpy as np
import re
import os
from google.colab import drive
from sklearn import preprocessing, model_selection, svm, neighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
drive.mount('/content/drive')

# %%
df = pd.read_csv('/content/drive/My Drive/File1.csv')

# %%
print(df.head(10))
df.describe()

# %%
gender_numerical = {"Male": 0, "Female": 1}
df['Gender'] = df['Gender'].map(gender_numerical)
display(df)

# %% [markdown]
# **Standardization**
# 
# MinMaxScaler() and StandardScaler() are the most commonly used scaling transforms for standardization.
# *Scaling / Standardization of data is preferably done after train_test_split as the scaling rules apply only for train data

# %%
scaler = MinMaxScaler()
df[["Age", "EstimatedSalary", "Gender", "Purchased"]] = scaler.fit_transform(df[["Age", "EstimatedSalary", "Gender", "Purchased"]])
print(df.head(10))
df.describe()

# %%
df = df.drop(['User ID'], 1)
display(df)

# %% [markdown]
# **Data Splitting**

# %%
X = df.drop(['Purchased'], 1)
y = df['Purchased']
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, test_size = 0.15)

# %% [markdown]
# **SVM** **Classiifer**

# %%
clf = svm.SVC(kernel='rbf', gamma='auto')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(confusion_matrix(y_test, pred))
accuracy = accuracy_score(y_test, pred)
print(accuracy)

# %% [markdown]
# KNN Classifier

# %%
clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(confusion_matrix(y_test, pred))
accuracy = accuracy_score(y_test, pred)
print(accuracy)

# %% [markdown]
# **Suitability of Supervised and Unsupervised Machine
# Learning Classifiers**
# 
# In the given problem statement, use of SVM and KNN Classifier is advised. Both of the said Classifiers are predominantly associated with Supervised Machine Learning as they deal with labelled data.
# 
# As a rule of thumb, if the given data is labelled, we tend to gravitate towards using Supervised models. However in some cases, it can be considered wise to apply unsupervised clustering models such as the K-means algorithm. This approach is especially crucial in problems statement where the given 'k' classes are unable to justify the expected accuracy hinting that the actual number of classes could be more/less. In such cases, unsupervised algorithms can be used to define the number of classes followed by supervised algorithms for prediction.

# %%



