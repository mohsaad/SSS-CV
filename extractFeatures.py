import numpy as np
import cv2
import time


camera = cv2.VideoCapture(0)
orb = cv2.ORB()

while(camera.isOpened()):
    t = time.time()
    print(time.time() - t)
    
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    '''     HARRIS CORNERS:
    corner = cv2.cornerHarris(gray, 2, 3, 0.04)
    frame[corner>.01*corner.max()] = [0, 255, 0]
    '''
    
    '''     ORB
    kp = orb.detect(gray,None)
    
    kp, des = orb.compute(gray, kp)
    
    img2 = cv2.drawKeypoints(gray, kp, color=(0,255,0), flags=0)
    '''
    
    cv2.imshow('Test', img2)
