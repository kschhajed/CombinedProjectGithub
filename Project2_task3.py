"""
Morphology Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with commonly used morphology
binary image processing techniques. Use the proper combination of the four commonly used morphology operations, 
i.e. erosion, dilation, open and close, to remove noises and extract boundary of a binary image. 
Specifically, you are given a binary image with noises for your testing, which is named 'task3.png'.  
Note that different binary image might be used when grading your code. 

You are required to write programs to: 
(i) implement four commonly used morphology operations: erosion, dilation, open and close. 
    The stucturing element (SE) should be a 3x3 square of all 1's for all the operations.
(ii) remove noises in task3.png using proper combination of the above morphology operations. 
(iii) extract the boundaries of the objects in denoised binary image 
      using proper combination of the above morphology operations. 
Hint: 
• Zero-padding is needed before morphology operations. 

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy libraries, HOWEVER, 
you are NOT allowed to use any functions or APIs directly related to morphology operations.
Please implement erosion, dilation, open and close operations ON YOUR OWN.
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np

filter = np.ones((3,3), dtype = np.int64)
k_h = filter.shape[0]//2
k_w = filter.shape[1]//2

def padding(img):
    padded_img = np.pad(img, pad_width=1)
    return padded_img

def morph_erode(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return erode_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology erosion on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """
   
    padded_img = padding(img)
    erode_img = np.empty(img.shape, dtype = int)
    padded_img = padded_img//255

    for i in range(k_h, padded_img.shape[0] - k_h):
        for j in range(k_w, padded_img.shape[1] - k_w):
            true_val = filter == (padded_img[i - k_h :i + k_h + 1,j - k_w:j + k_w + 1])
            if (true_val.all() == True):
                erode_img[i-k_h,j-k_w] = 1
            else:
                erode_img[i-k_h,j-k_w] = 0
    #print("Erosion:", erode_img.shape)
    erode_img = erode_img * 255
    erode_img = erode_img.astype('uint8')
    return erode_img


def morph_dilate(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return dilate_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology dilation on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    padded_img = padding(img)
    dilate_img = np.empty(img.shape, dtype = int)
    padded_img = padded_img//255

    for i in range(k_h, padded_img.shape[0] - k_h):
        for j in range(k_w, padded_img.shape[1] - k_w):
            true_val = filter == (padded_img[i - k_h :i + k_h + 1,j - k_w:j + k_w + 1])
            if (true_val.any() == True):
                dilate_img[i-k_h,j-k_w] = 1
            else:
                dilate_img[i-k_h,j-k_w] = 0
    #print("Dilation:",dilate_img.shape)
    dilate_img = dilate_img * 255
    dilate_img = dilate_img.astype('uint8')
    return dilate_img


def morph_open(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return open_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology opening on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """
    interim = morph_erode(img)
    open_img = morph_dilate(interim)
    open_img = open_img.astype('uint8')
    return open_img


def morph_close(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return close_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology closing on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """
    interim = morph_dilate(img)
    close_img = morph_erode(interim)
    close_img = close_img.astype('uint8')
    return close_img


def denoise(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Remove noises from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """
    interim = morph_open(img)
    denoise_img = morph_close(interim)
    denoise_img = denoise_img.astype('uint8')
    return denoise_img


def boundary(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Extract boundaries from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """
    interim = morph_erode(img)
    bound_img = img - interim
    bound_img = bound_img.astype('uint8')
    return bound_img


if __name__ == "__main__":
    img = imread('task3.png', IMREAD_GRAYSCALE)
    denoise_img = denoise(img)
    imwrite('results/task3_denoise.jpg', denoise_img)
    bound_img = boundary(denoise_img)
    imwrite('results/task3_boundary.jpg', bound_img)
