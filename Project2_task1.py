"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random

from numpy.testing._private.utils import nulp_diff
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """
    inliers = []
    sift = cv2.SIFT_create(nfeatures=1000)

    kp_left, des_left = sift.detectAndCompute(left_img,None)
    kp_left_n = [ i.pt for i in kp_left]

    kp_right, des_right = sift.detectAndCompute(right_img,None)
    kp_right_n = [ i.pt for i in kp_right]

    kp_left_n = np.array(kp_left_n)
    kp_right_n = np.array(kp_right_n)

    dist_n = []
    print("\nGetting 2 best matching keypoints (Pi, Qm, Qn)")
    for i in range(kp_left_n.shape[0]):
        dist = []
        for j in range(kp_right_n.shape[0]):
            dist.append([np.linalg.norm(des_left[i,:] - des_right[j,:]),kp_right_n[j] ])
        dist = sorted(dist,key= lambda x:x[0])
        dist_n.append([kp_left_n[i],dist[0][1],dist[0][0],dist[1][1],dist[1][0]])

    print("Finding and storing good matches in good_match[]")
    good_match = []
    for i in range(len(dist_n)):
        if(dist_n[i][2] < 0.75*dist_n[i][4]):
            good_match.append(dist_n[i][0:3])

    init_H = []
    t = 10 #threshold value

    print("Starting implementation of RANSAC algorithm")
    for k in range(0,5000):
        random.shuffle(good_match)
        H = computeHomography(good_match[:4])
        remaining = good_match[4:]
        s = []

        for i in range(len(remaining)):
            q = [remaining[i][1][0], remaining[i][1][1],1]
            preal = [remaining[i][0][0], remaining[i][0][1]]
            q = np.array(q)
            q = q.reshape(3,1)
            p = H@q
            if(p[2] != 0):
                p = p/p[2]
            p = p[:2]
            if ( np.linalg.norm(np.array([preal]).T - p) < t):
                s.append([[preal[0],preal[1]],[q[0][0],q[1][0]]])
        if ((k == 0) or (len(s) > len(inliers))):
            inliers = s[:]
            
    final_H = computeHomography(inliers)
    print("No. of inliers: ",len(inliers))
    print("Final Homography matrix computed using all inliers:\n ",final_H)

    final_H = np.array(final_H, dtype = "float32")
    corners = right_img.shape
    corner = np.array([[0,0],[corners[1],0],[0,corners[0]], [corners[1],corners[0]]], dtype = "float32")
    corner = np.array([corner])
    corner_new = cv2.perspectiveTransform(corner,final_H)
    corner_new = np.array(corner_new)
    print("New corners after multiplying existing corners with Homography matrix\n:",corner_new)
    minxy = np.min(np.min(corner_new,axis=0),axis=0)
    minx =0
    miny =0
    Hnew =0
    maxxy =0
    maxx =0
    maxy =0
    if (minxy[0]<0 or minxy[1]<0):
        if(minxy[0]<0):
            minx = np.abs(minxy[0])
            minx = np.int(minx)
        if(minxy[1]<0):
            miny = np.abs(minxy[1])
            miny = np.int(miny)
    transform = np.array([[1, 0 , minx],[0, 1, miny],[0, 0, 1]])
    Hnew = transform@final_H
    print("Transformation matrix\n:",transform)
    #print("H\n:", Hnew)
    corner_new = cv2.perspectiveTransform(corner,Hnew) 
    #print("Cornernew2\n",corner_new)   
    maxxy = np.max(np.max(corner_new,axis=0),axis=0)
    maxx = np.int(maxxy[0])
    maxy = np.int(maxxy[1])

    right_img_out = cv2.warpPerspective(right_img,Hnew, (maxx,maxy))   

    right_img_out[miny:miny + left_img.shape[0],
               minx:minx + left_img.shape[1]] = left_img
    result_img = right_img_out[0:maxy, minx:maxx] 
    return result_img

#Function to compute Homography matrix H using SVD
def computeHomography(init_H):

    A = np.empty((0,9))
    for i in range(0,len(init_H)):
        A = np.append(A ,np.array([[init_H[i][1][0], init_H[i][1][1], 1, 0, 0, 0, 
        -init_H[i][0][0]*init_H[i][1][0], -init_H[i][0][0]*init_H[i][1][1], -init_H[i][0][0] ],
            [0, 0, 0, init_H[i][1][0], init_H[i][1][1], 1, -init_H[i][0][1]*init_H[i][1][0], 
            -init_H[i][0][1]*init_H[i][1][1],  -init_H[i][0][1] ]]),axis = 0)

    #Applying SVD
    u, s, Vt = np.linalg.svd(A)  
    H_cap = Vt[-1:]
    H_cap = H_cap.reshape(3,3)
    if (H_cap[2][2] != 0):
        H_cap = H_cap/H_cap[2][2]
    return H_cap

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)
