###############
##Design the function "findRotMat" to  return 
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ 
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz 
#It is ok to add other functions if you need
###############

import numpy as np
import cv2

def createRz(angle):
    angle = angle * np.pi/180
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
    return Rz

def createRx(angle):
    angle = angle * np.pi/180
    Rx = np.array([[1, 0, 0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]])
    return Rx

def findRotMat(alpha, beta, gamma):
    t = createRz(gamma) @ createRx(beta) @ createRz(alpha)
    t2 = createRz(-alpha) @ createRx(-beta) @ createRz(-gamma)
    return t, t2

if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 60
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
    print(rotMat1)
    print(rotMat2)