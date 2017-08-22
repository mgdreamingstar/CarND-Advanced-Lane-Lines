import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
import pickle

# Constants

pipeline_camera_cal = glob.glob(r'D:\Github\CarND-Advanced-Lane-Lines\camera_cal\*.jpg')[0]
pipeline_thresh_combine = cv2.imread(r'test_images\test1.jpg')


# Helper Functions

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        x = 1
        y = 0
    elif orient == 'y':
        x = 0
        y = 1
    sobelxy = cv2.Sobel(gray, cv2.CV_64F, x, y)
    abs_sobelxy = np.absolute(sobelxy)
    scaled_sobel = np.int8(255 * abs_sobelxy / np.max(abs_sobelxy))
    sxbinary = np.zeros_like(scaled_sobel)
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobelxy = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    scaled_sobelxy = np.uint8(255 * mag_sobelxy / np.max(mag_sobelxy))
    thres_min = mag_thresh[0]
    thres_max = mag_thresh[1]
    sbinary = np.zeros_like(scaled_sobelxy)
    sbinary[(scaled_sobelxy >= thres_min) & (scaled_sobelxy <= thres_max)] = 1
    return sbinary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_gradient = np.arctan2(abs_sobely, abs_sobelx)
    binary = np.zeros_like(dir_gradient)
    min = thresh[0]
    max = thresh[1]
    binary[(dir_gradient >= min) & (dir_gradient <= max)] = 1

    return binary


def undistortion5(img_path):
    objp = np.zeros((5 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:5].T.reshape(-1, 2)
    imgpoints_un = []
    objpoints_un = []

    # img = mpimg.imread(img)
    origin = cv2.imread(img_path)
    # out_img = np.zeros_like(img_undist)
    gray = cv2.cvtColor(origin, cv2.COLOR_RGB2GRAY)  # convert to grayscale picture
    # plt.imshow(gray,cmap='gray')
    ret, corners = cv2.findChessboardCorners(gray, (9, 5), None)  # find imgpoints

    if ret:
        imgpoints_un.append(corners)
        objpoints_un.append(objp)
        # print(imgpoints_un)
        # img = cv2.drawChessboardCorners(img, (9,6), corners, ret) # draw imgpoints on image
        img_size = (origin.shape[1], origin.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_un, imgpoints_un, img_size, None, None)
        out_img = cv2.undistort(origin, mtx, dist, None, mtx)
        return out_img, mtx, dist


def undistortion6(img_path):
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    imgpoints_un = []
    objpoints_un = []

    # img = mpimg.imread(img)
    origin = cv2.imread(img_path)
    # out_img = np.zeros_like(img_undist)
    gray = cv2.cvtColor(origin, cv2.COLOR_RGB2GRAY)  # convert to grayscale picture
    # plt.imshow(gray,cmap='gray')
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)  # find imgpoints

    if ret:
        imgpoints_un.append(corners)
        objpoints_un.append(objp)
        # print(imgpoints_un)
        # img = cv2.drawChessboardCorners(img, (9,6), corners, ret) # draw imgpoints on image
        img_size = (origin.shape[1], origin.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_un, imgpoints_un, img_size, None, None)
        out_img = cv2.undistort(origin, mtx, dist, None, mtx)
        return out_img, mtx, dist


def undistortion5and6(image):
    try:
        undist_image, mtx, dist = undistortion5(image)
    except:
        undist_image, mtx, dist = undistortion6(image)
    return undist_image, mtx, dist


def hls_binary(hls_image):
    # hls_image = pipeline_thresh_combine
    hls = cv2.cvtColor(hls_image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(hls_image, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 25
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return color_binary, combined_binary


def warp(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[720, 470],  # top right
         [1050, 680],  # bottom right
         [260, 680],  # bottom left
         [565, 470]])  # top left

    dst = np.float32(
        [[1100, 0],
         [1100, 700],
         [260, 700],
         [260, 0]])
    '''
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

    dst = np.float32(
        [[(img_size[0] / 4), 10],
        [(img_size[0] / 4), img_size[1] - 10],
        [(img_size[0] * 3 / 4), img_size[1] - 10],
        [(img_size[0] * 3 / 4), 10]])
    '''
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, src, dst, M, Minv


# FIXME: OUTPUT LESS \u21A5

def lanes_finding(image, margin=30):
    # fit_image = cv2.imread(r'.\output_images\perspective_trans.jpg')
    if image.ndim == 2:
        fit_image = image
    else:
        fit_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    histogram = np.sum(fit_image, axis=0)

    # fit_image = cv2.imread(r'.\output_images\perspective_trans.jpg') # 读入更改视角的图片
    # fit_image = cv2.cvtColor(fit_image, cv2.COLOR_BGR2GRAY) # 改为单通道
    out_img = np.dstack((fit_image, fit_image, fit_image)) * 255

    midpoint = np.int(histogram.shape[0] / 2)  # shape[0] 是 y 轴
    leftx_base = np.argmax(histogram[:midpoint])  # 左边的峰值
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint  # 右边的峰值

    nwindows = 9
    window_height = np.int(fit_image.shape[0] / nwindows)

    nonzero = fit_image.nonzero()  # 所有非零点的坐标
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base  # 开始搜索的点
    rightx_current = rightx_base

    # margin = 100
    minpix = 50

    left_lane_inds = []  # 逻辑值数组
    right_lane_inds = []

    for window in range(nwindows):  # window = 0 - 8
        win_y_low = fit_image.shape[0] - (window + 1) * window_height
        win_y_high = fit_image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)  # 左边的方框
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)  # 右边的方框
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]  # 左边方框中点的个数
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]  # 右边方框中点的个数

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position 找到了合适的位置
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))  # 新的搜索点，下一个方框 X 方向的中心
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)  # 拉直
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]  # 左边点的 x 坐标
    lefty = nonzeroy[left_lane_inds]  # 左边点的 y 坐标
    rightx = nonzerox[right_lane_inds]  # 右
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)  # y 的 2 次多项式
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, fit_image.shape[0] - 1, fit_image.shape[0])  # y 坐标
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]  # x 左边的曲线坐标
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]  # x 右边的曲线坐标

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]  # 左边在方框中的点涂红
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]  # 蓝
    return out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, midpoint, leftx_base, rightx_base, leftx, rightx, lefty, righty, left_fitx, right_fitx, ploty, left_fit, right_fit


# FIXME OUTPUT LESS


def curvature(lefty, leftx, righty, rightx, ploty):
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
    return left_curverad, right_curverad


# lane_warped=pers_warped, Minv=Minv_w, left=left_fitx, right=right_fitx, y=ploty
def project_back(origin_image, lane_warped, Minv, left, right, y):
    warp_zero = np.zeros_like(lane_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # color_warp = warp_zero
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left, y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right, y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (origin_image.shape[1], origin_image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(origin_image, 1, newwarp, 0.3, 0)
    return result


def process_image(image):
    '''
    undist_image, mtx, dist = undistortion5and6(image)
    color_binary, combined_binary = hls_binary(hls_image)
    arped, src, dst, M, Minv = warp(img)
    out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, midpoint, leftx_base, rightx_base, leftx, rightx, lefty, righty, left_fitx, right_fitx, ploty, left_fit, right_fit = lanes_finding(image, margin=30)
    left_curverad, right_curverad = curvature(lefty = lefty, leftx = leftx, righty = righty, rightx = rightx)
    result = project_back(origin_image,lane_warped = pers_warped, Minv = Minv_w, left = left_fitx, right = right_fitx, y = ploty)

    image flow:

    pipeline_camera_cal = glob.glob(r'D:\Github\CarND-Advanced-Lane-Lines\camera_cal\*.jpg')[0]
    pipeline_thresh_combine = cv2.imread(r'test_images\test1.jpg')

    image = glob.glob(r'D:\Github\CarND-Advanced-Lane-Lines\camera_cal\*.jpg')[0] # pipeline_camera_cal
    camera_cal_dir = os.path.abspath(r'.\output_images\distortion-corrected.jpg')
    '''
    # undist_image, mtx, dist = undistortion5and6(image)  # mtx and dist not used.
    color_binary, combined_binary = hls_binary(image)
    warped, src, dst, M, Minv = warp(combined_binary)
    out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, midpoint, leftx_base, rightx_base, leftx, rightx, lefty, righty, left_fitx, right_fitx, ploty, left_fit, right_fit = lanes_finding(warped, margin=30)
    left_curverad, right_curverad = curvature(lefty=lefty, leftx=leftx, righty=righty, rightx=rightx)
    result = project_back(image, lane_warped=warped, Minv=Minv, left=left_fitx, right=right_fitx, y=ploty)
    return result

# %% test case
# camera cal
camera_cal_path = glob.glob(r'D:\Github\CarND-Advanced-Lane-Lines\camera_cal\*.jpg')[0]
undist_image, mtx, dist = undistortion5and6(camera_cal_path)
image = cv2.imread(r'test_images\test1.jpg')
image_undist = cv2.undistort(image, mtx, dist, None, mtx)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# hls binary
color_binary, combined_binary = hls_binary(image_undist)
plt.imshow(combined_binary, cmap='gray')

# warp image
warped, src, dst, M, Minv = warp(combined_binary)
plt.imshow(warped, cmap='gray')

print(warped.shape)
print(warped.ndim)

# lane finding
out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, midpoint, leftx_base, rightx_base, leftx, rightx, lefty, righty, left_fitx, right_fitx, ploty, left_fit, right_fit = lanes_finding(warped, margin=30)
plt.imshow(out_img,cmap='gray')

# curvature

left_curverad, right_curverad = curvature(lefty=lefty, leftx=leftx, righty=righty, rightx=rightx, ploty=ploty)
print('left curvature is {} m, right curvature is {} m'.format(left_curverad,right_curverad))

# project back

abc = project_back(image, lane_warped=warped, Minv=Minv, left=left_fitx, right=right_fitx, y=ploty)
plt.imshow(cv2.cvtColor(abc,cv2.COLOR_BGR2RGB))
