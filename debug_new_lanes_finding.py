import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

def warp(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
            [[720,470], # top right
             [1050,680], # bottom right
             [260,680], # bottom left
             [565,470]]) # top left

    dst = np.float32(
            [[1100,0],
             [1100,700],
             [260,700],
             [260,0]])
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


def LuvLab_binary(LuvLab_image):
    # hls_image = pipeline_thresh_combine
    Luv = cv2.cvtColor(LuvLab_image, cv2.COLOR_BGR2Luv)
    Lab = cv2.cvtColor(LuvLab_image, cv2.COLOR_BGR2Lab)

    L_channel = Luv[:,:,0]
    b_channel = Lab[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(LuvLab_image, cv2.COLOR_BGR2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 25
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    L_thresh_min = 235
    L_thresh_max = 255
    b_thresh_min = 170
    b_thresh_max = 255
    L_binary = np.zeros_like(L_channel)
    b_binary = np.zeros_like(b_channel)
    all_binary = np.zeros_like(L_channel)
    L_binary[(L_channel >= L_thresh_min) & (L_channel <= L_thresh_max)] = 1
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1
    all_binary[(L_binary == 1) | (b_binary == 1)] = 1
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, all_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(all_binary == 1) | (sxbinary == 1)] = 1
    return color_binary, combined_binary

def lanes_finding(image, margin=100):
    # fit_image = cv2.imread(r'.\output_images\perspective_trans.jpg')
    if image.ndim == 2:
        fit_image = image
    else:
        fit_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    histogram = np.sum(fit_image,axis=0)

    # fit_image = cv2.imread(r'.\output_images\perspective_trans.jpg') # 读入更改视角的图片
    # fit_image = cv2.cvtColor(fit_image, cv2.COLOR_BGR2GRAY) # 改为单通道
    out_img = np.dstack((fit_image,fit_image,fit_image))*255

    midpoint = np.int(histogram.shape[0]/2) # shape[0] 是 y 轴
    leftx_base = np.argmax(histogram[:midpoint]) # 左边的峰值
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # 右边的峰值

    nwindows = 9
    window_height = np.int(fit_image.shape[0]/nwindows)

    nonzero = fit_image.nonzero() # 所有非零点的坐标
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base # 开始搜索的点
    rightx_current = rightx_base

    # margin = 100
    minpix = 40

    left_lane_inds = [] # 逻辑值数组
    right_lane_inds = []

    for window in range(nwindows): # window = 0 - 8
        win_y_low = fit_image.shape[0] - (window+1)*window_height
        win_y_high = fit_image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 4) # 左边的方框
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 4) # 右边的方框
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0] # 左边方框中点的个数
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0] # 右边方框中点的个数

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position 找到了合适的位置
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds])) # 新的搜索点，下一个方框 X 方向的中心
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) # 拉直
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds] # 左边点的 x 坐标
    lefty = nonzeroy[left_lane_inds] # 左边点的 y 坐标
    rightx = nonzerox[right_lane_inds] # 右
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2) # y 的 2 次多项式
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, fit_image.shape[0]-1, fit_image.shape[0]) # y 坐标
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] # x 左边的曲线坐标
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] # x 右边的曲线坐标

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0] # 左边在方框中的点涂红
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 255, 255] # 蓝
    return out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, midpoint, leftx_base, rightx_base, leftx, rightx, lefty, righty, left_fitx, right_fitx, ploty, left_fit, right_fit

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.bias = None
        #base point
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

def new_lanes_finding(image, margin=80):
    '''
    introduce Line class.
    '''
    # fit_image = cv2.imread(r'.\output_images\perspective_trans.jpg')
    if image.ndim == 2:
        fit_image = image
    else:
        fit_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    histogram = np.sum(fit_image,axis=0)

    # fit_image = cv2.imread(r'.\output_images\perspective_trans.jpg') # 读入更改视角的图片
    # fit_image = cv2.cvtColor(fit_image, cv2.COLOR_BGR2GRAY) # 改为单通道
    out_img = np.dstack((fit_image,fit_image,fit_image))*255

    midpoint = np.int(histogram.shape[0]/2) # shape[0] 是 y 轴
    leftx_base = np.argmax(histogram[:midpoint]) # 左边的峰值
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # 右边的峰值

    nwindows = 9
    window_height = np.int(fit_image.shape[0]/nwindows)

    nonzero = fit_image.nonzero() # 所有非零点的坐标
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base # 开始搜索的点
    rightx_current = rightx_base

    # margin = 100
    minpix = 40

    left_lane_inds = [] # 逻辑值数组
    right_lane_inds = []

    for window in range(nwindows): # window = 0 - 8
        win_y_low = fit_image.shape[0] - (window+1)*window_height
        win_y_high = fit_image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 4) # 左边的方框
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 4) # 右边的方框
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0] # 左边方框中点的个数
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0] # 右边方框中点的个数

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position 找到了合适的位置
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds])) # 新的搜索点，下一个方框 X 方向的中心
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) # 拉直
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds] # 左边点的 x 坐标
    lefty = nonzeroy[left_lane_inds] # 左边点的 y 坐标
    rightx = nonzerox[right_lane_inds] # 右
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2) # y 的 2 次多项式
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, fit_image.shape[0]-1, fit_image.shape[0]) # y 坐标
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] # x 左边的曲线坐标
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] # x 右边的曲线坐标

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0] # 左边在方框中的点涂红
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255] # 蓝

    left_lane = Line()
    right_lane = Line()

    # left_lane.detected = True
    left_lane.recent_xfitted.append(np.array(left_fitx)) # list of points on fitted line
    left_lane.current_fit.append(left_fit) # np.array of fitted coefficients
    left_lane.bias = (midpoint - leftx_base) * 3.7/700
    left_lane.line_base_pos = leftx_base
    left_lane.diffs = left_lane.current_fit[-2] - left_lane.current_fit[-1]
    left_lane.allx = leftx # pixels
    left_lane.ally = lefty

    # right_lane.detected = True
    right_lane.recent_xfitted.append(np.array(right_fitx))
    right_lane.current_fit.append(right_fit)
    right_lane.bias = (midpoint - rightx_base) * 3.7/700
    right_lane.line_base_pos = rightx_base
    right_lane.diffs = right_lane.current_fit[-2] - right_lane.current_fit[-1]
    right_lane.allx = rightx
    right_lane.ally = righty

    return out_img, left_lane, right_lane


image_frame = cv2.imread('image_frame.png')
plt.imshow(image_frame)

warped, src, dst, M, Minv = warp(image_frame)
color_binary, combined_binary = LuvLab_binary(warped)

# %% lanes_finding
out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, midpoint, leftx_base, rightx_base, leftx, rightx, lefty, righty, left_fitx, right_fitx, ploty, left_fit, right_fit = lanes_finding(combined_binary)

plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')

# %% new_lanes_finding
out_img, left_lane, right_lane = new_lanes_finding(combined_binary, margin=80)

# 24th second debug on new_lanes_finding
clip = VideoFileClip(R'.\Videos\project_video.mp4')
image_frame = clip.get_frame(24)
# cv2.imwrite('24th_second.png',image_frame)
warped, src, dst, M, Minv = warp(image_frame)
color_binary, combined_binary = LuvLab_binary(warped)
out_img, left_lane, right_lane = new_lanes_finding(combined_binary, margin=80)
plt.imshow(warped)
