# %%
import numpy as np
import pandas as pd
import cv2 as cv2
from google.colab.patches import cv2_imshow
from skimage import io
from PIL import Image
import matplotlib.pylab as plt

#chttps://scikit-image.org/docs/dev/api/skimage.data.html

# %%


# %% [markdown]
# Low Pass Filters

# %%
from skimage import data
img=io.imread('lena_sp_noise.jpeg')
#img = data.coins()# Reading the image camers from the std dataset of skimage
io.imshow(img)
plt.title('Original image')

# %%


# %% [markdown]
# Homogenous Filters

# %%
kernel=np.ones([5,5])/25 # Homogeneous is an everaging filter. hence is smoothens and blurs the 

dst=cv2.filter2D(img,-1,kernel) #2d convolution
io.imshow(dst)
plt.title('Homogeneous filter using filter2D function')

# %%
plt.subplot(1,2,1),io.imshow(img), plt.title('Original Image')
plt.subplot(1,2,2),io.imshow(dst),plt.title('Output of Homogeneous filter')

blur=cv2.blur(img,(5,5))
io.imshow(blur)
plt.title('Homogeneous filter using blur method')

# %%


# %% [markdown]
# Gaussian Blur Filter

# %%
gblur=cv2.GaussianBlur(img,(5,5),1)
io.imshow(gblur)
plt.title('Gaussian blur filter')

plt.subplot(1,2,1),plt.imshow(dst,'gray'), plt.title('Homogeneous')
plt.subplot(1,2,2),plt.imshow(gblur,'gray'),plt.title('Gaussian blur')

# %%


# %% [markdown]
# Bilateral Filter

# %%
bblur=cv2.bilateralFilter(img,5,75,75)
io.imshow(bblur)
plt.title('Bilateral filter')

# %%


# %% [markdown]
# Median Blur Filter

# %%
mblur=cv2.medianBlur(img,5)
io.imshow(mblur)
plt.title('Median blur filter')

# %%


# %% [markdown]
# High Pass Filter

# %%
from skimage import data
img=data.coins()
io.imshow(img)

# %%


# %% [markdown]
# Laplacian Filter

# %%
lap=cv2.Laplacian(img, cv2.CV_64F)
# cv_64f is data type 64 bit float which can deal with negatives
#To convert it back to unsigned int
lap=np.uint8(np.absolute(lap))
io.imshow(lap)
plt.title('laplacian filter')

# %%


# %% [markdown]
# Sobel Operator

# %%
img1=data.checkerboard()
io.imshow(img1)

# %%
img1=data.checkerboard()
sobelx=cv2.Sobel(img1, cv2.CV_64F,1,0)# 1 0 for derivative in x direction
# cv_64f is data type 64 bit float which can deal with negatives
sobely=cv2.Sobel(img1, cv2.CV_64F,0,1)# 0 1 for derivative in y direction
#To convert it back to unsigned int
sobelx=np.uint8(np.absolute(sobelx))
sobely=np.uint8(np.absolute(sobely))

io.imshow(sobely)
plt.title('Sobel y filter')

# %%
io.imshow(sobelx)
plt.title('Sobel x filter')

# %%
sobelcombined_cheaker=cv2.bitwise_or(sobelx,sobely)
io.imshow(sobelcombined_cheaker)
plt.title('Sobel x and y filter combined')

# %%
sobelx=cv2.Sobel(img, cv2.CV_64F,1,0)# 1 0 for derivative in x direction
# cv_64f is data type 64 bit float which can deal with negatives
sobely=cv2.Sobel(img, cv2.CV_64F,0,1)# 0 1 for derivative in y direction
#To convert it back to unsigned int
sobelx=np.uint8(np.absolute(sobelx))
sobely=np.uint8(np.absolute(sobely))

# %%
io.imshow(sobely)
plt.title('Sobel y filter')

# %%
io.imshow(sobelx)
plt.title('Sobel x filter')

# %%
sobelcombined_coins=cv2.bitwise_or(sobelx,sobely)
io.imshow(sobelcombined_coins)
plt.title('Sobel x and y filter combined')

# %%
import seaborn as sns
sns.set()

# %%
plt.figure(figsize=(15,15))

plt.subplot(1,2,1),plt.imshow(lap ,'gray' ), plt.title('laplacian filter')
plt.subplot(1,2,2),plt.imshow(sobelcombined_coins,'gray'),plt.title('sobelcombined_coins')

# %%
from skimage import data
from skimage import exposure
img=io.imread('https://cdn.zeebiz.com/sites/default/files/styles/zeebiz_850x478/public/2018/11/14/59887-mutual-fund-reuters.jpg?itok=LT3xg6SO')
#img = data.moon()

# %%
io.imshow(img)

# %%


# %% [markdown]
# Image Enhancement Techniques
# 
# Gamma Correction

# %%
gamma_c= exposure.adjust_gamma(img,3)

plt.subplot(1,2,1),plt.imshow(img,'gray'), plt.title('Original')
plt.subplot(1,2,2),plt.imshow(gamma_c,'gray'),plt.title('Gamma correction')

# %%


# %% [markdown]
# Log Correction

# %%
logarithmic_corrected = exposure.adjust_log(img, 0.4)

plt.subplot(1,2,1),plt.imshow(img,'gray'), plt.title('Original')
plt.subplot(1,2,2),plt.imshow(logarithmic_corrected,'gray'),plt.title('Log correction')

# %%


# %% [markdown]
# Histogram Equalization

# %%
img = data.moon()
img1 = data.moon()
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist_full)

# %%
cv2.equalizeHist(img,img)
hist_full1 = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist_full1)

# %%
plt.subplot(1,2,1),plt.plot(hist_full), plt.title(' Original Histogram')
plt.subplot(1,2,2),plt.plot(hist_full1),plt.title('Equalised Histogram')

# %%
plt.subplot(1,2,1),plt.imshow(img1,'gray'), plt.title('Original Image')
plt.subplot(1,2,2),plt.imshow(img,'gray'),plt.title('Hist Equalised Image')

# %%



