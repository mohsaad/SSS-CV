import cv2
import numpy as np
from matplotlib import pyplot as plt

src = cv2.imread('pic.png')

#cv2.split is apparently very slow, so if we want to do this real time we probably want to use numpy indexing
#b, g, r = cv2.split('src')
#also I got a type error using cv2.split idk why.

red = src[:,:,2]
green = src[:,:,1]
blue = src[:,:,0]

plt.subplot(232),plt.imshow(src,'gray'),plt.title('Original')
plt.subplot(234),plt.imshow(blue,'gray'),plt.title('Blue')
plt.subplot(235),plt.imshow(green,'gray'),plt.title('Green')
plt.subplot(236),plt.imshow(red,'gray'),plt.title('Red')

plt.show()