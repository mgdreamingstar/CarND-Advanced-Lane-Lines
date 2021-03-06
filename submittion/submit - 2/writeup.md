## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* (F) Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* (F) Apply a distortion correction to raw images.
* (F) Use color transforms, gradients, etc., to create a thresholded binary image.
* (F) Apply a perspective transform to rectify binary image ("birds-eye view").
* (F) Detect lane pixels and fit to find the lane boundary.
* () Determine the curvature of the lane and vehicle position with respect to center.
* (F) Warp the detected lane boundaries back onto the original image.
* () Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/writeup_undistorted_image.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/combined.jpg "Binary Example 1"
[image4]: ./output_images/writeup_combined_hls_image.jpg "Binary Example 2"
[image5]: ./output_images/writeup_perspective_transformation.jpg "Perspective Transformation"
[image6]: ./output_images/writeup_fit_lane_lines.jpg "Fit lane lines"
[image7]: ./output_images/project_back.jpg "Project Back"
[video1]: ./project_video.mp4 "Video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd and 3rd code cell of the IPython notebook located in "./Advance-Lane-Lines.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints_un` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints_un` will be appended with the (x, y) pixel position of each of the corners in the **image plane** with each successful chessboard detection.

I then used the output `objpoints_un` and `imgpoints_un` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function. In order to obtain the result, I set two functions to respectively address cases which has either 5 or 6 rows on the chessboard:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

First, I used a combination of sobel gradient, magnatitude and direction thresholds to generate a binary image. The code for this step is contained in the 4th and 5th code cell of the IPython notebook located in "./Advance-Lane-Lines.ipynb".

Here's an example of my output for this step. The result isn't very good.
![alt text][image3]

Then, I use sobel_x and s channel to get a binary image. The code for this step is contained in the 6th code cell of the IPython notebook located in "./Advance-Lane-Lines.ipynb".

Here's an example of my output for this step. On this image below, the left part is a image in which green and blue area comes from sobel_x and s channel threshold. The right part is the binary output. The result is pretty good.
![alt text][image4]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the 8th code cell of the IPython notebook).  The `warp()` function takes as inputs an image (`img`).  I chose the following source and destination points on the images:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 720, 470     | 1100, 0        |
| 1050, 680      | 1100,700     |
| 260,680     | 260, 700   |
| 565,470     | 260, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

This step is coming from the course using sliding box to detect pixels belong to the left or right lane. After the pixels are found, use a 2nd order polynomial to fit each lane.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in 11th and 12th code cell of the IPython notebook.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in in 15th code cell of the IPython notebook in the function `project_back()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

##### Issues:

Firstly, the pictures provided for camera calibration don't have save rows, but they have either 5 or 6 rows. So I use a `try...except...` method to address this issue.

Secondly, when detecting lanes, the width of boxes is quite importent. As it will take in outlier pixels, so I set a relatively small width to address this issue.

##### Where will my pipeline likely fail?

The right white lane is disconnected, so the detection sometimes is uncertain. After testing on the challenge_video.mp4, I find on these situations the pipeline may fail:

1. There is another line on road which isn't lane line.
2. There is shadow on road.
3. The color of lanes are fading.

##### What could I do to make it more robust?

I think to
1. have several different color space threshold to detect the lanes will make it more robust.
2. take the direction of lanes into account will make it more robust.
3. look-ahead when the detection is uncertain and smoothing on the last n findings will make it more robust.
