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
    print("Done Matching")
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
    print(startr,endr, startc,endc)
    image = image[startr:endr, startc:endc]
    if(startr ==1 and endr==0 and startc==1 and endc ==0):
        flag = True
    return image , flag

start_time = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
	help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to the output image")
args = vars(ap.parse_args())


print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])),key=natural_keys)
#print(imagePaths)
images = []


print("stitching ", imagePaths[0], "and", imagePaths[1])
img1 = cv2.imread(imagePaths[0])#cv2.imread("images/DSC00247.JPG")
img2 = cv2.imread(imagePaths[1])#cv2.imread("images/DSC00267.JPG")
pts1, pts2, matches1to2, kp1, kp2 = genSIFTMatchPairs(img1, img2)
(matches, H, status) = getHomography(kp1, kp2, pts1, pts2, matches1to2, 1)
width = img1.shape[1] + img2.shape[1]
height = img1.shape[0] + img2.shape[0]
result = cv2.warpPerspective(img1, H, (width, height))
img2 = cv2.copyMakeBorder(img2, 0, img2.shape[0], 0, img2.shape[1], cv2.BORDER_CONSTANT, None, value=0)
#result[0:img2.shape[0], 0:img2.shape[1]] = img2
#img2 = cv2.copyMakeBorder(img1, 0, img1.shape[0], 0, img1.shape[1], cv2.BORDER_CONSTANT, None, value=0)
'''
width = 2*img1.shape[1] + img2.shape[1]
height = 2*img1.shape[0] + img2.shape[0]
print("image 1 padded",height-img1.shape[0],width - img1.shape[1])
img1 = cv2.copyMakeBorder(img1, height-img1.shape[0], height-img1.shape[0], width - img1.shape[1], width - img1.shape[1], cv2.BORDER_CONSTANT, None, value=0)
print("to shape" , img1.shape)
result = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))

# result = cv2.warpPerspective(img1, H, (width, height))
print("image 2 padded",height-img2.shape[0],width - img2.shape[1])
img2 = cv2.copyMakeBorder(img2, height-img2.shape[0], height-img2.shape[0], width - img2.shape[1], width - img2.shape[1], cv2.BORDER_CONSTANT, None, value=0)
print("to shape" , img2.shape)
'''
dst_mask = 1 - binary_mask(result)
dst_mask = np.stack((dst_mask,) * 3, -1)
print(img2.shape, dst_mask.shape, result.shape)
result = np.multiply(img2, dst_mask) + result
result , flag= trim_image(result)
flag = False
cv2.imwrite('output.jpg', result)
print("--- first two images took %s seconds ---" % (time.time() - start_time))
for i in range(2,len(imagePaths)):
    s = 'output' + str(i - 1) + ".jpg"
    print("stitching",s," and", imagePaths[i])
    if(i==2):
        img2 = cv2.imread("output.jpg")
    else:
        img2 = cv2.imread(s)
    ori = img2
    img1 = cv2.imread(imagePaths[i])
    pts1, pts2, matches1to2, kp1, kp2 = genSIFTMatchPairs(img1, img2)
    (matches, H, status) = getHomography(kp1, kp2, pts1, pts2, matches1to2, 1)
    width = img1.shape[1] + img2.shape[1]
    height = img1.shape[0] + img2.shape[0]
    result = cv2.warpPerspective(img1, H, (width, height))
    img2 = cv2.copyMakeBorder(img2, 0, height - img2.shape[0], 0, width - img2.shape[1], cv2.BORDER_CONSTANT, None, value=0)
    dst_mask = 1 - binary_mask(result)
    dst_mask = np.stack((dst_mask,) * 3, -1)
    print(img2.shape, dst_mask.shape, result.shape)
    result = np.multiply(img2, dst_mask) + result
    print("result is", result.shape)
    result, flag = trim_image(result)

    #resize to smaller
    if flag:
        print("rejected",imagePaths[i])
        result = ori
    flag=False
    if(result.shape[0]>5000 or result.shape[1] > 5000):
        print("resizing output to optimize time")
        result = np.array(result, dtype='float32')
        result=cv2.resize(result,None,fx=.75,fy=.75)
    s = 'output' + str(i) + ".jpg"
    cv2.imwrite(s, result)
    print("--- took %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

'''width = img1.shape[1] + img2.shape[1]
height = img1.shape[0] + img2.shape[0]
result = cv2.warpPerspective(img1, H, (width, height))
'''
'''
(width1, height1) = result.size
(width2, height2) = img2.size

result_width = width1 + width2
result_height = max(height1, height2)

result = Image.new('RGB', (result_width, result_height))
result.paste(im=result, box=(0, 0))
result.paste(im=img2, box=(width1, 0))
'''
'''
img2 = cv2.copyMakeBorder(img2,0, img2.shape[0], 0, img2.shape[1], cv2.BORDER_CONSTANT, None, value = 0)
dst_mask = 1 - binary_mask(result)
dst_mask = np.stack((result,))
dst_mask = dst_mask[0]
print(img2.shape, dst_mask.shape, result.shape)
result = np.multiply(img2, dst_mask) + result
'''

'''
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(c)
result = result[y:y + h, x:x + w]
'''


'''
width = img1.shape[1] + img2.shape[1]
height = img1.shape[0] + img2.shape[0]
result = cv2.warpPerspective(img1, H, (width, height))

img1 = cv2.copyMakeBorder(img2, 0, img2.shape[0], 0, img2.shape[1], cv2.BORDER_CONSTANT, None, value = 0)
print(result.shape, img1.shape)
result = cv2.addWeighted(result,0.5,img1,0.5,0)
'''
'''
result[0:img2.shape[0], 0:img2.shape[1]] = img2
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(c)
result = result[y:y + h, x:x + w]
'''


