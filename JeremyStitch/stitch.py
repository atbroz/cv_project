from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time
import winsound

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
	help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to the output image")
ap.add_argument("-m", "--mode", type=str, required=True,
	help="mode of stitcher")
args = vars(ap.parse_args())
start=time.time()

print("loading images")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	if (image.shape[0] * image.shape[1] > 8081920):  # around 4k
		print('resizing',imagePath, 'to save memory and time')
		while(image.shape[0] * image.shape[1] > 8081920):
			image = cv2.resize(image, (None), fx=.95, fy=.95)
	images.append(image)

print('loaded',len(images),'images')
print('loading took',time.time()-start)
stitchstart=time.time()

print("stitching")
stitcher = cv2.createStitcher(1) if imutils.is_cv3() else cv2.Stitcher_create(1)
(status, stitched) = stitcher.stitch(images)
print(status)
cv2.imwrite(args["output"], stitched)
print("stitching exited with status",status)
print('DONE. stitching took',time.time()-stitchstart,'The whole program took',time.time()-start)
frequency = 1000  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)




