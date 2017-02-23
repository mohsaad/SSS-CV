import cv2
import numpy as np
from matplotlib import pyplot as plt

src = cv2.imread('Horizon-2.jpg')

#cv2.split is apparently very slow, so if we want to do this real time we probably want to use numpy indexing
#b, g, r = cv2.split('src')
#also I got a type error using cv2.split idk why.

red = src[:,:,2]
green = src[:,:,1]
blue = src[:,:,0]



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
openRed = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
openGreen = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
openBlue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)

finalRed = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
finalGreen = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
finalBlue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)

final = cv2.merge([finalBlue, finalGreen, finalRed])

plt.subplot(332),plt.imshow(src,'gray'),plt.title('Original')
plt.subplot(334),plt.imshow(blue,'gray'),plt.title('Blue')
plt.subplot(335),plt.imshow(green,'gray'),plt.title('Green')
plt.subplot(336),plt.imshow(red,'gray'),plt.title('Red')
plt.subplot(337),plt.imshow(finalBlue,'gray'),plt.title('smooth blue')
plt.subplot(338),plt.imshow(finalGreen,'gray'),plt.title('smooth green')
plt.subplot(339),plt.imshow(finalRed,'gray'),plt.title('smooth red')

plt.show()
