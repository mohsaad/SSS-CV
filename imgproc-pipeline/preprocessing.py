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
main method, use to test.
'''

def main():
	pass

'''
If this python script is run by itself, run the main method.
'''
if __name__ == '__main__':
	main()
