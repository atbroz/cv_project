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


def load_image(filename):
    img = np.asarray(Image.open(filename))
    img = img.astype("float32")/255.0
    return img

def show_image(img, title):
    fig = plt.figure()
    fig.set_size_inches(18, 10)             # You can adjust the size of the displayed figure
    plt.imshow(img)
    plt.title(title)



def kmeans_segment(K, img, ker):

    img = cv2.GaussianBlur(img, ker, 0)
    img_convert = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    vectorized = img_convert.reshape((-1, 3))   # converts image to (MXN, 3) size
    vectorized = np.float32(vectorized)         # openCV requires float for k-means

    attempts = 20
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)

    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    # colors = np.unique(result_image, axis=1).reshape(-1,3)
    # mask = np.zeros(result_image.shape)

    # for c in range(colors.shape[0]):
    #  for i in range(result_image.shape[0]):
    #    for j in range(result_image.shape[1]):
    #      if np.array_equal(result_image[i][j], colors[c]):
    #        mask[i][j] = [255-colors[c][0], 255-colors[c][1], 255-colors[c][2]]

    return result_image



def means_shift(img, ker, w):
    # w is used in the estimate_bandwidth function as the quantile value and must be between 0 and 1
    # it is a placeholder variable that relates to the window size

    # filter to reduce noise
    img = cv2.GaussianBlur(img, ker, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # flatten the image
    flat_image = img.reshape((-1, 3))
    flat_image = np.float32(flat_image)
    # print('flat image: \n', flat_image)

    # meanshift
    bandwidth = estimate_bandwidth(flat_image, quantile=w, n_samples=3000)
    ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
    ms.fit(flat_image)
    labeled = ms.labels_

    # get number of segments
    segments = np.unique(labeled)
    print('Number of segments: ', segments.shape[0])

    # get the average color of each segment
    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total / count
    avg = np.uint8(avg)

    # cast the labeled image into the corresponding average color
    res = avg[labeled]
    result = res.reshape((img.shape))

    return result


def segmented_outputs(img, cluster, k=0.25):
    #k is resize factor
    colors = np.unique(cluster.reshape(-1, cluster.shape[2]), axis=0)
    mask = [np.array(img) for i in range(colors.shape[0])]
    for color in range(colors.shape[0]):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if not (cluster[i][j] == colors[color]).all():
                    mask[color][i, j] = 0

    final_mask = cv2.resize(mask[0], (0, 0), None, k, k)
    for ii in range(colors.shape[0] - 1):
        final_mask = np.hstack((final_mask, cv2.resize(mask[ii + 1], (0, 0), None, k, k)))

    return final_mask
