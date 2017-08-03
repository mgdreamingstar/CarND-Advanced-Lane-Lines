import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        x = 1
        y = 0
    elif orient == 'y':
        x = 0
        y = 1
    sobelxy = cv2.Sobel(gray,cv2.CV_64F,x,y)
    abs_sobelxy = np.absolute(sobelxy)
    scaled_sobel = np.int8(255*abs_sobelxy/np.max(abs_sobelxy))
    sxbinary = np.zeros_like(scaled_sobel)
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    mag_sobelxy = np.sqrt(np.power(sobelx,2)+np.power(sobely,2))
    scaled_sobelxy = np.uint8(255*mag_sobelxy/np.max(mag_sobelxy))
    thres_min = mag_thresh[0]
    thres_max = mag_thresh[1]
    sbinary = np.zeros_like(scaled_sobelxy)
    sbinary[(scaled_sobelxy >= thres_min) & (scaled_sobelxy <= thres_max)] = 1
    return sbinary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_gradient = np.arctan2(abs_sobely, abs_sobelx)
    binary = np.zeros_like(dir_gradient)
    min = thresh[0]
    max = thresh[1]
    binary[(dir_gradient >=  min) & (dir_gradient <= max)] = 1

    return binary
