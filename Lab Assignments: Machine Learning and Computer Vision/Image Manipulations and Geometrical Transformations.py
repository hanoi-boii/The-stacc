# %%
import numpy as np
import pandas as pd
import cv2 as cv
from google.colab.patches import cv2_imshow
from skimage import io
from PIL import Image
import matplotlib.pylab as plt
#chttps://scikit-image.org/docs/dev/api/skimage.data.html

# %% [markdown]
# Experiment 1: Image Manipulations and Geometrical Transformations

# %%
from skimage import data
camera = data.camera() # Reading the image camera from the std dataset of skimage

# %%
camera

# %%
io.imshow(camera)

# %%


# %% [markdown]
# Retrieving the geometry of the image and the number of pixels:

# %%
camera.shape

# %%
camera.size

# %%
type(camera)

# %% [markdown]
# Retrieving statistical information about image intensity values:

# %%
camera.min(), camera.max()

# %%
camera.mean()

# %%
io.imshow(camera)

# %%
camera

# %%


# %% [markdown]
# Image manupulation

# %%
 # Set the first 20 rows and all column to "black" (0)
 camera[1:20,:] = 0
 
  # Set the next 20 rows and all column to "white" (255)
 camera[21:40,:] = 255
 io.imshow(camera)

# %%
 camera[:,1:20] = 0
 camera[:,21:40] = 255
 io.imshow(camera)

# %%


# %% [markdown]
# Masking (indexing with masks of booleans)

# %%
mask = camera < 87
mask

# %%
mask = camera < 87
 # Set to "white" (255) the pixels where mask is True
camera[mask] = 255
io.imshow(camera)

# %%
mask

# %%


# %%
moon=data.moon()
io.imshow(moon)

# %%
moon.shape

# %%
#resize moon
from skimage.transform import resize
moon_resized = resize(moon,[600,600])
io.imshow(moon_resized)

# %%


# %% [markdown]
# Concatenate two images

# %%
newimage1=np.concatenate((moon,moon), axis=1)
io.imshow(newimage1)

# %%
newimage2=np.concatenate((newimage1,newimage1), axis=0)
io.imshow(newimage2)

# %%
camera = data.camera() # Reading the image camera from the std dataset of skimage

newimage2=np.concatenate((camera[1:150,:],camera[350:512,:]), axis=0)
io.imshow(newimage2)

# %%


# %% [markdown]
# CROPPING OF IMAGE

# %%
io.imshow(camera[1:260,261:512])# cropping of FIRST QUADRANT OF THE IMAGE

# %%


# %%
 img = data.astronaut()

io.imshow(img)

# %%
img

# %%
top_left = img[:100, :100]
io.imshow(top_left)

# %%


# %% [markdown]
# GEOMETRICAL OPERATIONS

# %%
#Geometrical operations
from skimage import color
from skimage.transform import rescale, resize, downscale_local_mean, rotate

image = color.rgb2gray(img)#Convert color image to gray scale image

# %%
io.imshow(image)

# %%
image_rescaled = rescale(image, 0.25)
io.imshow(image_rescaled)

# %%
image_rescaled = rescale(image, 4)
io.imshow(image_rescaled)

# %%
image_rescaled = rescale(image, 0.25)
io.imshow(image_rescaled)

# %%


# %%
#image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4))
#io.imshow(image_resized)
image_resized = resize(image, (125,250))
io.imshow(image_resized)

# %%
image_downscaled = downscale_local_mean(image, (4,4))
io.imshow(image_downscaled)

# %%
image_rotate=rotate(image,-45)
io.imshow(image_rotate)

# %%



