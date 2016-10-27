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
	edges = edge_detection(smooth, 25, 150)
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

def hough(edges):
	lines = cv2.HoughLines(edges.astype('uint8'),1,np.pi/180,200)
	# print(lines.shape)

	if lines is None:
		return edges

	for i in range(0, lines.shape[0]):
		rho = lines[i,0,0]
		theta = lines[i,0,1]
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		# slope = float(y2-y1)/(x2 - x1)

		cv2.line(edges,(x1,y1),(x2,y2),128,2)
	return edges


def pipeline(image):
	# image = read_image(pathname)
	colors = split_and_recombine(image)
	image = merge(colors)
	image = dilation(image, (3,3))
	image = hough(image)

	return image

'''
main method, use to test.
'''

def main():
	import sys
	cap = cv2.VideoCapture(sys.argv[1])
	#image = pipeline(sys.argv[1])
	#image[:,:] *= 255

	if cap.isOpened():
		print "Video opened succesfully"
	else : 
		print "Video did not open succesfully"

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == -1:
			break
		image = pipeline(frame)
		cv2.imshow("frame", frame)
		cv2.imshow("cv", image)
		cv2.waitKey(1)

	cap.release()
	#cv2.imwrite(sys.argv[2], image)

'''
If this python script is run by itself, run the main method.
'''
if __name__ == '__main__':
	main()
