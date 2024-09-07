# %% [markdown]
# **AI&DA18 Batch 1 Lab Exam Name: Atharva M. Kulkarni GR No: 11810384**
# 
# Q9: Design and Implement different ANN models to predict the median house value using the
# following dataset. (Use median_house_price column as a y label)
# Dataset Link: https://drive.google.com/file/d/1_yg6u5zqusErB3DSRXfW18IST2ButJXp/view?usp=sharing
# 
# A. Create different models for following optimizers
# 1. momentum
# 2. rmsprop
# 3. adagrad
# 4. adam
# 
# B. Split data using 80:20 split and random_state=26
# 
# C. Train every model using above data and store their epochs wise loss (use min 10 epochs)
# 
# D. Plot the loss against number_of_epochs for every optimizer and conclude which optimizer converges faster.

# %%
import pandas as pd
import numpy as np
import re
import os
from google.colab import drive
drive.mount('/content/drive')
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

# %%
df = pd.read_csv('/content/drive/My Drive/Q2_housing_data.csv')

# %%
display(df)

# %% [markdown]
# **Data Preprocessing**
# 
# a. Dealing with NaNs
# 
# b. Dealing with Categorical Data
# 
# c. Train-Test Split
# 
# d. Data Scaling

# %%
print(df.columns)
print(df.isnull().sum())

# %%
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
print(df.isnull().sum())

# %%
ocean_proximity_set = set(df['ocean_proximity'])
print(ocean_proximity_set)

proximity_types = {"ISLAND": 0, "<1H OCEAN": 1, "NEAR BAY": 2, "INLAND": 3, "NEAR OCEAN": 4}
df['ocean_proximity'] = df['ocean_proximity'].map(proximity_types)
display(df)

# %%
scaler = MinMaxScaler()
df[["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_house_value", "ocean_proximity"]] = scaler.fit_transform(df[["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_house_value", "ocean_proximity"]])
display(df)
df.describe()

# %%
#X = df.drop(['median_house_value'], 1)
#y = df['median_house_value']
#X = np.array(X)
#y = np.array(y)

#X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, test_size = 0.2, random_state = 26, shuffle = True)
#with open('/content/drive/MyDrive/Lab VP/housing_price_train_test.pkl', 'wb') as f:
#  pickle.dump((X_train, X_test, y_train, y_test), f)

with open('/content/drive/MyDrive/Lab VP/housing_price_train_test.pkl', 'rb') as f:
  X_train, X_test, y_train, y_test = pickle.load(f)

# %% [markdown]
# **SGD Optimizer with Momentum = 0.2**

# %%
model_SGD = Sequential()
model_SGD.add(Dense(32, kernel_initializer='uniform', input_dim = X_train.shape[1], activation='relu'))
model_SGD.add(Dropout(0.1))
model_SGD.add(Dense(64, kernel_initializer='uniform',activation='relu'))
model_SGD.add(Dropout(0.1))
model_SGD.add(Dense(32, kernel_initializer='uniform',activation='relu'))
model_SGD.add(Dropout(0.2))
model_SGD.add(Dense(1, kernel_initializer='uniform',activation='linear'))

tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.2, name="SGD")
model_SGD.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = 'SGD', metrics = ['mean_squared_error'])
model_SGD.summary()

# %%
model_history_SGD = model_SGD.fit(X_train, y_train, epochs = 30, batch_size = 8, validation_split = 0.1)

# %%
train_loss_SGD = model_history_SGD.history['loss']
val_loss_SGD = model_history_SGD.history['val_loss']
epoch_range = range(1, 31)
plt.plot(epoch_range, train_loss_SGD, 'g', label = "Training Loss")
plt.plot(epoch_range, val_loss_SGD, 'b', label = "Validation Loss")
plt.title('SGD with Momentum=0.2 losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# **RMSProp Optimizer**

# %%
model_rmsprop = Sequential()
model_rmsprop.add(Dense(32, kernel_initializer='uniform', input_dim = X_train.shape[1], activation='relu'))
model_rmsprop.add(Dropout(0.1))
model_rmsprop.add(Dense(64, kernel_initializer='uniform',activation='relu'))
model_rmsprop.add(Dropout(0.1))
model_rmsprop.add(Dense(32, kernel_initializer='uniform',activation='relu'))
model_rmsprop.add(Dropout(0.2))
model_rmsprop.add(Dense(1, kernel_initializer='uniform',activation='linear'))

tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.0, name="RMSprop")
model_rmsprop.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = 'RMSprop', metrics = ['mean_squared_error'])
model_rmsprop.summary()

# %%
model_history_rmsprop = model_rmsprop.fit(X_train, y_train, epochs = 30, batch_size = 8, validation_split = 0.1)

# %%
train_loss_rmsprop = model_history_rmsprop.history['loss']
val_loss_rmsprop = model_history_rmsprop.history['val_loss']
epoch_range = range(1, 31)
plt.plot(epoch_range, train_loss_rmsprop, 'g', label = "Training Loss")
plt.plot(epoch_range, val_loss_rmsprop, 'b', label = "Validation Loss")
plt.title('RMSProp with Momentum=0.0 losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# **AdaGrad Optimizer**

# %%
model_adagrad = Sequential()
model_adagrad.add(Dense(32, kernel_initializer='uniform', input_dim = X_train.shape[1], activation='relu'))
model_adagrad.add(Dropout(0.1))
model_adagrad.add(Dense(64, kernel_initializer='uniform',activation='relu'))
model_adagrad.add(Dropout(0.1))
model_adagrad.add(Dense(32, kernel_initializer='uniform',activation='relu'))
model_adagrad.add(Dropout(0.2))
model_adagrad.add(Dense(1, kernel_initializer='uniform',activation='linear'))

tf.keras.optimizers.Adagrad(learning_rate=0.001, name="Adagrad")
model_adagrad.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = 'Adagrad', metrics = ['mean_squared_error'])
model_adagrad.summary()

# %%
model_history_adagrad = model_adagrad.fit(X_train, y_train, epochs = 30, batch_size = 8, validation_split = 0.1)

# %%
train_loss_adagrad = model_history_adagrad.history['loss']
val_loss_adagrad = model_history_adagrad.history['val_loss']
epoch_range = range(1, 31)
plt.plot(epoch_range, train_loss_adagrad, 'g', label = "Training Loss")
plt.plot(epoch_range, val_loss_adagrad, 'b', label = "Validation Loss")
plt.title('AdaGrad with losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# **Adam Optimizer**

# %%
model_adam = Sequential()
model_adam.add(Dense(32, kernel_initializer='uniform', input_dim = X_train.shape[1], activation='relu'))
model_adam.add(Dropout(0.1))
model_adam.add(Dense(64, kernel_initializer='uniform',activation='relu'))
model_adam.add(Dropout(0.1))
model_adam.add(Dense(32, kernel_initializer='uniform',activation='relu'))
model_adam.add(Dropout(0.2))
model_adam.add(Dense(1, kernel_initializer='uniform',activation='linear'))

tf.keras.optimizers.Adam(learning_rate=0.001, name="Adam")
model_adam.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = 'Adam', metrics = ['mean_squared_error'])
model_adam.summary()

# %%
model_history_adam = model_adam.fit(X_train, y_train, epochs = 30, batch_size = 8, validation_split = 0.1)

# %%
train_loss_adam = model_history_adam.history['loss']
val_loss_adam = model_history_adam.history['val_loss']
epoch_range = range(1, 31)
plt.plot(epoch_range, train_loss_adam, 'g', label = "Training Loss")
plt.plot(epoch_range, val_loss_adam, 'b', label = "Validation Loss")
plt.title('Adam (default beta values) with losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# **Comparing the Different Optimizers**
# 
# During the said activity, we've applied the following Optimizers to our ANN model:
# 
# a) SGD with momentum=0.2
# 
# b) RMSProp
# 
# c) AdaGrad
# 
# d) Adam
# 
# On comparing the results, we can comment that for this particular problem statement, Adam and RMSProp show similar behavior and converge the quickest. AdaGrad starts with a high arbitrary state and as a result takes some more time to converge while SGD converges the slowest owing to it's relatively primitive nature.
# 

# %%
plt.plot(epoch_range, train_loss_SGD, 'r', label = "SGD Loss")
plt.plot(epoch_range, train_loss_rmsprop, 'g', label = "RMSProp Loss")
plt.plot(epoch_range, train_loss_adagrad, 'b', label = "Adagrad Loss")
plt.plot(epoch_range, train_loss_adam, 'k', label = "Adam Loss")
plt.title('Optimizers Comparision')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%



