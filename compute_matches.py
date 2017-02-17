#!/usr/bin/env python


import numpy as np
import cv2
down = cv2.imread("down.jpg", cv2.IMREAD_GRAYSCALE)
up = cv2.imread("up.jpg", cv2.IMREAD_GRAYSCALE)



#     HARRIS CORNERS:
corners = cv2.cornerHarris(down, 2, 3, 0.04)
corner_down = corners[corners > 0.01 * corners.max()]

corners = cv2.cornerHarris(up, 2, 3, 0.04)
corner_up = corners[corners > 0.01 * corners.max()]

max_corner_resp = corners > 0.01 * corners.max()
print(max_corner_resp.shape)
