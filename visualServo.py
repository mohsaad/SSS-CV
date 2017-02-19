import numpy as np
from numpy.linalg import inv
import cv

def VisualServo(x, y, z, a, b, c, d, e, f):
    L = np.matrix([[(1.0/z), 0.0, -(x/z), -x*y, (1.0 + x**2), -y], [0.0, (1.0/z), -(y/z), -(1.0 + y**2), x*y, x]])
    Lx = np.matrix([[(1.0/c), 0.0, -(a/c), -a*b, (1.0 + a**2), -b], [0.0, (1.0/c), -(b/c), -(1.0 + b**2), a*b, a]])
    Ly = np.matrix([[(1.0/f), 0.0, -(d/f), -d*e, (1.0 + d**2), -e], [0.0, (1.0/f), -(e/f), -(1.0 + e**2), d*e, d]])
    L = np.concatenate((L, Lx, Ly), axis=0)
    
    S = np.matrix([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0]])
   

        Lt = L.transpose()
        Lp = np.matmul(Lt, L)
        Linv = inv(Lp)
        Lp = np.matmul(Linv, Lt)
        Vc = np.matmul(Lp, S)


    return L, S, Vc

L, S, Vc = VisualServo(1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 9.0, 7.5, 8.45)
print(L, S, Vc)



