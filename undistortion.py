import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
%matplotlib inline
# # Read in the saved objpoints and imgpoints
# dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
# objpoints = dist_pickle["objpoints"]
# imgpoints = dist_pickle["imgpoints"]

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

camera_cal = os.path.abspath('D:\Github\CarND-Advanced-Lane-Lines\camera_cal')
imglist = glob.glob(camera_cal + '\*.jpg')

for idx, imgname in enumerate(imglist):
    img = mpimg.imread(imgname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # convert to grayscale picture

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)  # find imgpoints

    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)

        # img = cv2.drawChessboardCorners(img, (9,6), corners, ret) # draw imgpoints on image
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imshow('Undistorted', undist)
        cv2.waitKey(500)

cv2.destroyAllWindows()


# undistorted = cal_undistort(img, objpoints, imgpoints)
#
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(undistorted)
# ax2.set_title('Undistorted Image', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
