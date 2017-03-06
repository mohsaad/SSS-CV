#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
    # Also return the image if you'd like a copy
    return out


def import_images():
	#import images
	downorig = cv2.imread("../imgs/desk_1_1.png", cv2.IMREAD_GRAYSCALE)
	uporig = cv2.imread("../imgs/desk_1_2.png", cv2.IMREAD_GRAYSCALE)
	depthmap = cv2.imread("../imgs/desk_1_2_depth.png", cv2.IPL_DEPTH_16U)
	#resize images to 1/4 of size
	#height, width = uporig.shape
	#up = cv2.resize(uporig, (int(hei.ght * .25), int(width * .25)))
	#down = cv2.resize(downorig, (int(height * .25), int(width * .25)))
	return uporig, downorig, depthmap

def find_keypoints(up, down):
	#create the ORB object
	orb = cv2.ORB()

	#find keypoints and descriptors
	ft1, des1 = orb.detectAndCompute(down, None)
	ft2, des2 = orb.detectAndCompute(up, None)

	#BF Matching object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors and create 
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	up, down, depth = import_images()

	#filtered matches
	filtmat = []
	for mat in matches:
		#get indices
		index1 = mat.queryIdx
		index2 = mat.trainIdx

		#get positions
		(x1, y1) = ft1[index1].pt
		(x2, y2) = ft2[index2].pt

		#check depth
		z2 = depth[int(round(y2)), int(round(x2))]
		if z2 != 0 :
			filtmat.append(mat)
	return ft1, ft2, filtmat

def position_vector():
	posv = np.zeros((5,5))
	up, down, depth = import_images()
	kp1, kp2, matches = find_keypoints(up, down)

	#draw the matches
	'''
	cv2.imshow("out", drawMatches(up, kp1, down, kp2, matches))
	cv2.waitKey(0)
	'''
	for i in range(0, 5):
		#find the kp indices
		idx1 = matches[i].queryIdx
		idx2 = matches[i].trainIdx

		#get positions
		(u1, v1) = kp1[idx1].pt
		(u2, v2) = kp2[idx2].pt
		z2 = depth[int(round(v2)), int(round(u2))]

		#velocity
		udot = u2 - u1
		vdot = v2 - v1

		posv[i][0] = udot
		posv[i][1] = vdot
		posv[i][2] = u2
		posv[i][3] = v2
		posv[i][4] = z2
	return posv