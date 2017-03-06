#!/home/andrew/anaconda2/envs/SSS-CV/bin python

import numpy as np
from numpy.linalg import inv
import cv2
from compute_matches import position_vector

#This function takes a position vector and breaks it down into its individual components
'''
def SdotBreakdown(positionVector):
    s1 = positionVector[0]
    s2 = positionVector[1]
    return s1, s2
'''
#This function computes the image jacobian matrix given inputs x,y,z (u, v, z) and the feature value matrix S
def VisualServo(s1,s2,x,y,z):
    #focal height and width
    fy = 2.87
    fx = 3.58
    L = np.matrix([[(fx/z), 0.0, -(x/z), -x*y/fx , (fy**2 + x**2)/fy, -y], [0.0, (fy/z), -(y/z), -(fx**2 + y**2)/fx, x*y/fy, x]])
    S = np.matrix([[s1],[s2]])
    return L, S

#This function solves for camera velocity using the equation Vc = S * ((L^T * L)^-1 * L^T)
def SolveVisualServo(L,S):
    Lt = L.transpose()
    Lp = np.matmul(Lt, L)
    Linv = inv(Lp)
    Lp = np.matmul(Linv, Lt)
    Vc = np.matmul(Lp, S)
    return Vc

L = np.zeros((6,6))
S = np.zeros((6,1))
posv = position_vector()

for i in range(0,3):
    tempL, tempS = VisualServo(posv[i][0], posv[i][1], posv[i][2], posv[i][3], posv[i][4])
    L[2 * i : 2 * i + 2, :] = tempL
    S[2 * i : 2 * i + 2] = tempS
print("L", L)
print("S", S)
print("Vc", SolveVisualServo(L,S))
