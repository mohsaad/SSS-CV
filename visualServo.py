import numpy as np
from numpy.linalg import inv
import cv

#This function takes a position vector and breaks it down into its individual components
def SdotBreakdown(positionVector):
    s1 = positionVector[0]
    s2 = positionVector[1]
    return s1, s2

#This function computes the image jacobian matrix given inputs x,y,z (u, v, z) and the feature value matrix S
def VisualServo(x,y,z,s1,s2):
    L = np.matrix([[(1.0/z), 0.0, -(x/z), -x*y, (1.0 + x**2), -y], [0.0, (1.0/z), -(y/z), -(1.0 + y**2), x*y, x]])
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



