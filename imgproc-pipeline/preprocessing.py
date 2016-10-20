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
def morphological_smoothing(img):
	#5x5 elliptical shaped smoothing element
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	openimg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	return cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)

'''
Implements edge detection on current image.
'''
def edge_detection(img):
	sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
	return sobelX

'''
Dilation: dilates the image in order to make edges more visible.
'''
def dilation(img):
	#create a 5x5 circular element
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	return cv2.dilate(img,kernel,iterations = 1)


'''
The first part of the pipeline. Splits into R,G,B channels, performs
morphological smoothing, and edge detection, and dilation
'''
def process_channel(img):
	smooth = morphological_smoothing(img)
	edges = edge_detection(smooth)
	dilated = dilation(edges)

	return dilated



def split_and_recombine(img):
	# Use all three color channels
	colors = []
	for i in range(img.shape[2]):
		colors.append(process_channel(img[:,:,i]))
	
def merge(colors):
	row, cols, channels = colors.shape
	merged_image = colors[0]
	for i in range(1,channels):
		merged_image = cv2.bitwise_and(merged_image, colors[i])
	return merged_image


'''
main method, use to test.
'''

def main():
	pass

'''
If this python script is run by itself, run the main method.
'''
if __name__ == '__main__':
	main()
