###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER,findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):

    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #Capturing image and converting to grayscale
    img = imread(imgname)
    gray = cvtColor(img, COLOR_BGR2GRAY)
    # Finding the chess board corners and reshaping the array/matrix
    ret, corners = findChessboardCorners(gray, (4,9), None)
    a=[]
    b=[]
    corners = corners.reshape(36,2)

    if ret == True:
        corners2 = cornerSubPix(gray,corners, (4,9), (-1,-1), criteria)

    #Building World coordinates array and reshaping it in objpoints2 

    for x in range (40,0,-10):
        for z in range (40,0,-10):   
            a.append([x,0,z])

    for y in range (0,50,10):
        for z in range (40, 0, -10):
            b.append([0,y,z])

    objpoints2 = np.array((a+b))
    objpoints2 = objpoints2.reshape(36,3)

    #Building n*12 matrix for solving Ax=0, where n=72

    A = np.empty((0,12))

    for i in range (0,36):
          A = np.append(A , np.array([[objpoints2[i,0], objpoints2[i,1], objpoints2[i,2], 1, 
            0, 0, 0, 0, 
            -(corners[i,0]*objpoints2[i,0]), -(corners[i,0]*objpoints2[i,1]), -(corners[i,0]*objpoints2[i,2]), -(corners[i,0])],
            [0, 0, 0, 0,
            objpoints2[i,0], objpoints2[i,1], objpoints2[i,2], 1,
            -(corners[i,1]*objpoints2[i,0]), -(corners[i,1]*objpoints2[i,1]), -(corners[i,1]*objpoints2[i,2]), -(corners[i,1])]
            ] ),axis = 0)  

    #Applying SVD to find 
    u, s, Vt = np.linalg.svd(A)  
    x1 = Vt[-1:]

    M = x1.reshape(3,4)

    m3 = np.array(M[2,:3]).T

    #Finding lambda

    lam = 1/np.linalg.norm(m3)

    M = lam * M

    #Defining m1,m2,m3 in order to calculate f_x,f_y,o_x,o_y
    m1 = M[0,:3].T
    m2 = M[1,:3].T
    m3 = M[2,:3].T

    o_x = (m1.T) @ m3
    o_y = (m2.T) @ m3
    f_x = np.sqrt(((m1.T) @ m1) - np.square(o_x))
    f_y = np.sqrt(((m2.T) @ m2) - np.square(o_y))

    return [f_x, f_y, o_x, o_y], True

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)