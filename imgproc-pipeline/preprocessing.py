#!/usr/bin/env python

# Student Space Systems
# Computer Vision Group
# 
# 10/10/2016
# An implementation of the paper by Dusha et. al
# on horizon tracking and attitude determination

import numpy as np
import cv2


'''
read_image - reads an image inro our current memory.
'''
def read_image(pathname):
	return cv2.imread(pathname)


'''
Implements a morphological smoothing of the image.
First step in the preprocessing pipeline	
'''
def morphological_smoothing(img, kernel_size):
	#5x5 elliptical shaped smoothing element
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
	openimg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	return cv2.morphologyEx(openimg, cv2.MORPH_CLOSE, kernel)

'''
Implements edge detection on current image.
'''
def edge_detection(img, low, high):
	canny = cv2.Canny(img, low, high)
	return canny

'''
Dilation: dilates the image in order to make edges more visible.
'''
def dilation(img, kernel):
	#create a 5x5 circular element
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel)
	return cv2.dilate(img,kernel,iterations = 1)


'''
The first part of the pipeline. Splits into R,G,B channels, performs
morphological smoothing, and edge detection, and dilation
'''
def process_channel(img):
	smooth = morphological_smoothing(img, (3,3))
	edges = edge_detection(smooth, 50, 200)
	dilated = dilation(edges, (3,3))

	return dilated



def split_and_recombine(img):
	# Use all three color channels
	colors = np.zeros(img.shape)
	for i in range(img.shape[2]):
		colors[:,:,i] = (process_channel(img[:,:,i]))
	return colors
	
def merge(colors):
	rows, cols, channels = colors.shape
	merged_image = np.ones([rows, cols])*255
	for i in range(channels):
		merged_image[:,:] = cv2.bitwise_and(merged_image, colors[:,:,i])
	return merged_image


def pipeline(pathname):
	image = read_image(pathname)
	colors = split_and_recombine(image)
	masked_image = merge(colors)
	processed_image = dilation(masked_image, (3,3))

	return processed_image

'''
main method, use to test.
'''

def main():
	import sys
	image = pipeline(sys.argv[1])
	image[:,:] *= 255
	cv2.imwrite(sys.argv[2], image)

'''
If this python script is run by itself, run the main method.
'''
if __name__ == '__main__':
	main()
