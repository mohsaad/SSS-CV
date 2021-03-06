import numpy as np
import cv2
import sys
import time

t1 = time.time()

img = cv2.imread("../Horizon-2.jpg")
Z = img.reshape((img.shape[0] * img.shape[1],3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

print time.time() - t1

cv2.imwrite('res3.png',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()