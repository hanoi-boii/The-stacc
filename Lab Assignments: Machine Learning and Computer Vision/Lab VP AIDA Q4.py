# %% [markdown]
# **AI&DA18 Batch 1 Lab Exam Name: Atharva M. Kulkarni GR No: 11810384**
# 
# Q4: Design and Implement your own CNN model to predict the face is with a mask or not using thefollowing dataset. (Use “Train” directory for training images)
# Dataset Link: https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
# 
# A. Use Sufficient Number of Conv Layers, Activation functions and Loss functions
# 
# B. Train the model using 80% data and save trained model in .h5 format
# 
# C. Write a function to get output for a given input face image.

# %%
import numpy as np
import re
import os
from google.colab import drive
drive.mount('/content/drive')
import matplotlib.pyplot as plt
import cv2
from random import shuffle
from os import listdir
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from tensorflow import keras
from google.colab.patches import cv2_imshow
import pickle

# %%
DATA_DIR = '/content/drive/MyDrive/Face Mask Dataset/Train'
VAL_DIR = '/content/drive/MyDrive/Face Mask Dataset/Validation'
IMG_SIZE = 50
categories = ['WithMask', 'WithoutMask']

# %%
def create_train_data():
  training_data = []
  for category in categories:
    path = os.path.join(DATA_DIR, category)
    class_num = np.array(categories.index(category))
    for img in tqdm(listdir(path)):
      try:
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = np.array(cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)))
        training_data.append([new_array, class_num])
      except Exception as e:
        pass

  return training_data

# %%
def create_validation_data():
  val_data = []
  for category in categories:
    path = os.path.join(VAL_DIR, category)
    class_num = categories.index(category)
    for img in tqdm(listdir(path)):
      try:
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        val_data.append([new_array, class_num])
      except Exception as e:
        pass

  return val_data

# %%
train = create_train_data()
shuffle(train)

# %%
val = create_validation_data()
shuffle(val)

# %%
#X_train = []
#y_train = []
#for features, labels in train:
#  X_train.append(features)
#  y_train.append(labels)

#X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#y_train = np.array(y_train)
#X_train = X_train/255.0

#with open('/content/drive/MyDrive/Lab VP/mask_train_data.pkl', 'wb') as f:
#  pickle.dump((X_train, y_train), f)

with open('/content/drive/MyDrive/Lab VP/mask_train_data.pkl', 'rb') as f:
  X_train, y_train = pickle.load(f)

# %%
#X_val = []
#y_val = []
#for features, labels in val:
#  X_val.append(features)
#  y_val.append(labels)

#X_val = np.array(X_val).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#y_val = np.array(y_val)
#X_val = X_val/255.0

#with open('/content/drive/MyDrive/Lab VP/mask_val_data.pkl', 'wb') as f:
#  pickle.dump((X_val, y_val), f)

with open('/content/drive/MyDrive/Lab VP/mask_val_data.pkl', 'rb') as f:
  X_val, y_val = pickle.load(f)

# %%
model = Sequential()
model.add(Conv2D(256, (3,3), input_shape = X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

# %%
model_history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

# %%
train_loss = model_history.history['loss']
train_acc = model_history.history['accuracy']
epoch_range = range(1, 11)
plt.figure(figsize=(12, 4), dpi=80)
plt.subplot(1,2,1),plt.plot(train_acc), plt.title('Accuracy')
plt.subplot(1,2,2),plt.plot(train_loss),plt.title('Loss')

# %%
model.save('/content/drive/MyDrive/FaceMask.h5')

# %%
model = keras.models.load_model('/content/drive/MyDrive/FaceMask.h5')
val_loss, val_acc = model.evaluate(X_val, y_val)

# %%
def image_testing(img_path):
  img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  return new_array

# %%
#img_path = '/content/drive/MyDrive/Face Mask Dataset/Test/WithMask/Augmented_58_3749134.png'
#img_path = '/content/drive/MyDrive/Face Mask Dataset/Test/WithMask/Augmented_391_7745278.png'
#img_path = '/content/drive/MyDrive/Face Mask Dataset/Test/WithoutMask/3402.png'
#img_path = '/content/drive/MyDrive/Face Mask Dataset/Test/WithoutMask/420.png'

#img_path = '/content/drive/MyDrive/Face Mask Dataset/Test/WithoutMask/1177.png'
img_path = '/content/drive/MyDrive/Face Mask Dataset/Test/WithoutMask/4391.png'

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
cv2_imshow(img)

# %%
test_img = image_testing(img_path)
test_img = test_img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
prediction = model.predict(test_img)

print(prediction[0][0])
if prediction[0][0] == 0:
  print("Wearing a mask")
else:
  print("Not wearing a mask")


