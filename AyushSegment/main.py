import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# PIL is the Python Imaging Library
from PIL import Image


import pandas as pd
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_sample_image

from segment import *



print('Python version:', sys.version)
print('OpenCV version:', cv2.__version__)
print('NumPy version: ', np.__version__)
print('Pandas version:', pd.__version__)


img_og = cv2.imread('kentland_image.jpg')

kmeans = kmeans_segment(6, img_og, (13,13))



fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1,2,1)
ax1.imshow(cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB))
ax1.title.set_text('OG')
plt.axis('off')

ax2 = fig1.add_subplot(1,2,2)
ax2.imshow(kmeans)
ax2.title.set_text('kmeans')
plt.axis('off')
plt.show()


mshift = means_shift(img_og, (13,13), 0.15)

fig2 = plt.figure(2)
ax1 = fig2.add_subplot(1,2,1)
ax1.imshow(cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB))
ax1.title.set_text('OG')
plt.axis('off')

ax2 = fig2.add_subplot(1,2,2)
ax2.imshow(mshift)
ax2.title.set_text('meanshift')
plt.axis('off')
plt.show()


# To get the segmented outputs
out = segmented_outputs(img_og, mshift)
cv2.imshow('meanshift-segmented', out)
cv2.waitKey(0)







