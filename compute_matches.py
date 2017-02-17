#!/usr/bin/env python
import numpy as np
import cv2
import matplotlib.pyplot as plt


import numpy as np
import cv2

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

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
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 3)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 3)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 5)
    # Also return the image if you'd like a copy
    return out

#import images
downorig = cv2.imread("down.jpg", cv2.IMREAD_GRAYSCALE)
uporig = cv2.imread("up.jpg", cv2.IMREAD_GRAYSCALE)

height, width = uporig.shape
up = cv2.resize(uporig, (int(height * .25), int(width * .25)))
down = cv2.resize(downorig, (int(height * .25), int(width * .25)))

'''
#     HARRIS CORNERS:
corners = cv2.cornerHarris(down, 2, 3, 0.04)
corner_down = corners[corners > 0.01 * corners.max()]

corners = cv2.cornerHarris(up, 2, 3, 0.04)
corner_up = corners[corners > 0.01 * corners.max()]

max_corner_resp = corners > 0.01 * corners.max()
print(max_corner_resp.shape)
'''

#ORB
orb = cv2.ORB()

#keypoints/features
ft1, des1 = orb.detectAndCompute(down, None)
ft2, des2 = orb.detectAndCompute(up, None)

#draw features
downft = cv2.drawKeypoints(down, ft1, None, color=(0,255,0), flags=0)
upft = cv2.drawKeypoints(up, ft2, None, color=(0,255,0), flags=0)

'''
#draw the features
plt.imshow(downft),plt.show()
plt.imshow(upft),plt.show()
'''

#BF Matching object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
out = drawMatches(down,ft1,up,ft2,matches[:5])

cv2.imwrite('out.png', out)
'''
cv2.waitKey(0)
cv2.destroyWindow('Matched Features')
'''