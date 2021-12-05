#usage python3 main.py -i "path to images" -o output.png
import sys
import cv2
import numpy as np
import time
# PIL is the Python Imaging Library
import imutils
#from PIL import Image
#import Image
import argparse
from imutils import paths

#print(im1)
#cv2.imwrite('output.jpg', np.concatenate([im1, im2], axis=1))

def distance(xy1, xy2):
    (x1, y1), (x2, y2) = xy1, xy2
    return ((float(y2-y1))**2 + (float(x2-x1))**2)**0.5


def fake_image_corners(xy_sequence):
    """Get an approximation of image corners based on available data."""
    all_x, all_y = zip(*xy_sequence)
    min_x, max_x, min_y, max_y = min(all_x), max(all_x), min(all_y), max(all_y)
    d = dict()
    d['tl'] = min_x, min_y
    d['tr'] = max_x, min_y
    d['bl'] = min_x, max_y
    d['br'] = max_x, max_y
    return d


def corners(xy_sequence, image_corners):
    """Return a dict with the best point for each corner."""
    d = dict()
    d['tl'] = min(xy_sequence, key=lambda xy: distance(xy, image_corners['tl']))
    d['tr'] = min(xy_sequence, key=lambda xy: distance(xy, image_corners['tr']))
    d['bl'] = min(xy_sequence, key=lambda xy: distance(xy, image_corners['bl']))
    d['br'] = min(xy_sequence, key=lambda xy: distance(xy, image_corners['br']))
    return d

def genSIFTMatchPairs(img1, img2):
    print("SIFT")
    sift = cv2.SIFT_create()
    # kp = sift.detect(img1, None)
    # kp = cv2.SIFT(edgeThreshold=10)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    print("matching")
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)#L1 is slightly faster but L2 more accurate
    matches = bf.match(des1, des2)
    #print("Done Matching")
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.zeros((250, 2))
    pts2 = np.zeros((250, 2))
    for i in range(250):
        pts1[i, :] = kp1[matches[i].queryIdx].pt
        pts2[i, :] = kp2[matches[i].trainIdx].pt

    return pts1, pts2, matches[:250], kp1, kp2





#cv2.imwrite('output.jpg', matching_result)
def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    print("Homography")
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                         reprojThresh)
        #H = cv2.getPerspectiveTransform(ptsA, ptsB)
        #print(ptsA)
        #print()
        #print(ptsB)
        return (matches, H, status)
    else:
        return None
def binary_mask(img):
  mask = (img[:, :, 0] > 0) | (img[:, :, 1] > 0) | (img[:, :, 2] > 0)
  mask = mask.astype("int")
  return mask

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def trim_image(image):
    startr = 0
    endr = 0
    startc = 0
    endc = 0
    flag = False
    for r in range(image.shape[0]):
        if (np.sum(image[r]) != 0 and startr == 0):
            startr = r
        if (startr != 0 and np.sum(image[r]) == 0):
            endr = r
            break

    for r in range(image.shape[1]):
        if (np.sum(image[:, r]) != 0 and startc == 0):
            startc = r
        if (startc != 0 and np.sum(image[:, r]) == 0):
            endc = r
            break
    #print(startr,endr, startc,endc)
    image = image[startr:endr, startc:endc]
    if(startr ==1 and endr==0 and startc==1 and endc ==0):
        flag = True
    return image , flag

def stitch(img1, img2):
    #print(img1.shape,img2.shape)
    if(img2.shape[0] *img2.shape[1] > 8081920):#around 4k
        print('resizing to save memory and time')
        img2 = cv2.resize(img2,(None),fx=.75,fy=.75)
        img1 = cv2.resize(img1,(None), fx=.75, fy=.75)
        print('after resizing, images are', img1.shape, img2.shape)
    img2 = cv2.copyMakeBorder(img2, img1.shape[0], img1.shape[0], img1.shape[1], img1.shape[1], cv2.BORDER_CONSTANT, None, value=0)
    img1 = cv2.copyMakeBorder(img1, img1.shape[0], img1.shape[0], img1.shape[1], img1.shape[1], cv2.BORDER_CONSTANT, None, value=0)
    #print('after padding, images are',img1.shape,img2.shape)
    pts1, pts2, matches1to2, kp1, kp2 = genSIFTMatchPairs(img1, img2)
    (matches, H, status) = getHomography(kp1, kp2, pts1, pts2, matches1to2, 1)
    width = img1.shape[1] + img2.shape[1]
    height = img1.shape[0] + img2.shape[0]
    result = cv2.warpPerspective(img1, H, (width, height))
    img2 = cv2.copyMakeBorder(img2, 0, img1.shape[0], 0, img1.shape[1], cv2.BORDER_CONSTANT, None, value=0)
    dst_mask = 1 - binary_mask(result)
    dst_mask = np.stack((dst_mask,) * 3, -1)
    #print(img2.shape, dst_mask.shape, result.shape)
    result = np.multiply(img2, dst_mask) + result
    result, flag = trim_image(result)
    return result

start_time = time.time()
total_start_time = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
	help="path to input directory of images to stitch")
args = vars(ap.parse_args())


print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])),key=natural_keys)
#print(imagePaths)
print("found",len(imagePaths),"images")
images = []

'''
print("stitching ", imagePaths[0], "and", imagePaths[1])
img1 = cv2.imread(imagePaths[0])#cv2.imread("images/DSC00247.JPG")
img2 = cv2.imread(imagePaths[1])#cv2.imread("images/DSC00267.JPG")

result = stitch(img1,img2)
cv2.imwrite('tripleoutput.jpg', result)
print("--- first two images took %s seconds ---" % (time.time() - start_time))
'''
start_time = time.time()
for i in range(0,len(imagePaths),3):
    for j in range(3):
        if (i + j == len(imagePaths) - 1):
            break
        if(j==0):
            print("stitching", imagePaths[i + j], " and", imagePaths[i + j +1])
            img1 = cv2.imread(imagePaths[i + j + 1])
            img2 = cv2.imread(imagePaths[i + j])
            result = stitch(img1, img2)
            cv2.imwrite('tripleIntermediate.jpg', result)
            print("--- took %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
        else:
            print("stitching", 'tripleIntermediate.jpg', " and", imagePaths[i + j+1])
            img1 = cv2.imread(imagePaths[i + j+1])
            img2 = cv2.imread('tripleIntermediate.jpg')

            result = stitch(img1, img2)
            cv2.imwrite('tripleIntermediate.jpg', result)
            print("--- took %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
    if(i==0):
        cv2.imwrite('tripleoutput.jpg', result)
    else:
        print("stitching", 'tripleIntermediate.jpg', " and", 'tripleoutput.jpg')
        img1 = cv2.imread('tripleIntermediate.jpg')
        img2 = cv2.imread('tripleoutput.jpg')
        result = stitch(img1, img2)
        print("--- took %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        cv2.imwrite('tripleoutput.jpg', result)
