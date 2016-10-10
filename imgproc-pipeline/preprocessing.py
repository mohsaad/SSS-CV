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
	pass

'''
Implements edge detection on current image.
'''
def edge_detection(img):
	pass

'''
Dilation: dilates the image in order to make edges more visible.
'''
def dilation(img):
	pass


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
