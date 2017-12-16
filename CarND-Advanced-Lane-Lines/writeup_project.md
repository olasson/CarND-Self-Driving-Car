## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./my_images/calib_full_output.png "Calibration Output"
[image2]: ./my_images/undist_calib_1.png "Undistorted Calibration Image"
[image3]: ./my_images/undist_calib_2.png "Undistorted Test Image"
[image4]: ./my_images/combined_messy.png "Messy Thresholding"
[image5]: ./my_images/combined_clean.png "Clean Thresholding"
[image6]: ./my_images/birds_eye_transform.png "Birds-eye"
[image7]: ./my_images/src_and_dst.png "Src and Dst"
[image8]: ./my_images/slid_window_poly.png "Sliding window polynomial fit output"
[image9]: ./my_images/slid_window_prev_poly.png "Sliding window polynomial using previous fit output"
[image10]: ./my_images/text_and_lane.png "Text and lane Drawn"


[video1]: ./project_video_output.mp4 "Video"
[video2]: ./challenge_video_output.mp4 "Video"
[video3]: ./harder_challenge_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The calibration happens in code cells 11 and 12. I followed the steps in [This link](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html) for calibrating the camera. The functions that do most of the heavy lifing is the OpenCV functions `cv2.findChessBoardCorners` and `cv2.calibrateCamera()`. First, I made `obj_points_for_img` which contains 3D world coordinates (X, Y, Z) for the location of each checcboard corner for a single image. Furthermore, I made, based on the link above, the assumption that the chessboard is fixed, i. e Z = 0. The array `obj_points` contains object points for all calibration images. I did the same for the 2D pixel coordiantes with `img_points`, which containes all coners found by `cv2.findChessBoardCorners`. Finally, I simply fed `cv2.calibrateCamera()` with object and image points, and got back the camera calibration. As a visualization step, I used `cv2.drawChessboardCorners` for visualizing the calibration, see the output below. 

![alt text][image1]

After this, I used the distortion coefficients from the camera calibration to undistort a calibration image with `cv2.undistort()`, as seen below

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I used `cv2.undistort()` on a test image aswell. Here is the result

![alt text][image3]

The effects of the undistortion is most noticable when looking at the hood of the car. 

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I tried various comibnations of thresholding, which happens in cells 16-23. First, I tried to use the following threshold combination:

* Absolute value of sobel derivative gradient
* Magnitude of gradient
* Direction of gradient
* HLS colorspace L-channel
* HLS colorspace S-channel
* LAB colorspace B-channel

I used the "or" operator for combining them. The result can be seen here

![alt text][image4]

which I was not happy with, as it created way too much noise, which would probably create issues for the sliding window polyfit. I noticed that when using the "or" operator for combining thresholds, the gradient based thresholds tended to introduce alot of noise. The same was true for HLS colorspace S-channel thresholding when shadows were present in the image. Therefore, I opted for only using 

* HLS colorspace L-channel
* LAB colorspace B-channel

and the result can be seen here 

![alt text][image5]

which I was happy with. 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is found in the code cell number 3, and is called `warp()`. It is based on the OpenCv functions `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()`. It takes an image and src, dst points as inputs, and returns the warped image and the inverse transformation matrix for later use. The result of `warp()` can be seen below

![alt text][image6]

I chose to hardcode the src and dst points in the following manner, under the assumption that the camera position is more or less constant, and that the road is more or less flat. 

```python
src = np.float32([(570, 470), (720, 465), (260, 680), (1050, 685)])
dst = np.float32([(450, 0), (n_cols - 450, 0), (450, n_rows), (n_cols - 450, n_rows)])
```

I made frequent use of the helper function `show_src()` for making src and dst. 

Next, I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions `sliding_window_polyfit()` and `sliding_window_polyfit_using_prev_fit()` are found in code cells 5 and 6, and are mainly based on the Udacity lectures. I added an option for requesting an image of the result, amongst some other minor tweaks. Both uses a histogram based method, where they use the combined binary image created by my pipeline to identity the lane lines as "spikes" in the histogram. The function `sliding_window_polyfit()` computes a histogram for the bottom half of the image, and finds the bottommost x-position of the lange lines. These locations are identified based on the local maxima of the left and right halves of the histogram. The function allows the user to specify how many windows should be used, i.e the "resolution" for the detection. Each of these windows are centered on the midpoint of the pixels of the window below, and limits the area we need to search for the lane lines. The result from `sliding_window_polyfit()` can be seen below

![alt text][image8]

The next function `sliding_window_polyfit_using_prev_fit()` works in a similar way as `sliding_window_polyfit()`, but exploits a previously found fit through only searching for lane lines in an area described by the previous fits, see the image below 

![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

These functions are called `calculate_curvature()` and `calculate_center_distance()` and are found in code cell 7. 

The function `calculate_curvature()` is based on the udacity lectures, and uses conversion factors `ym_per_pix = 30/720, xm_per_pix = 3.7/700` to go from pixels to real world values. It fits a new polynomial to x,y in world space, and uses that to calculate the curvature according to the formula from the Udacity lectures. It calculates the curvature at the bottom position of the image (closest to the car). 

The function `calculate_center_distance()` calculates the intercepts of the left and right fits, and uses this to find the offset from the center of the lane. The fits in this case are calculated for the bottom image position (max y value). 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This step uses the functions `draw_lane()` and `draw_data()` found in code cell 8. Here is the result

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

* Link [Project Video Output](./project_video_output.mp4)

I also tried my implementation on the challenge videos, here are some links for the interested:
* Link [Challenge Video Output](./challenge_video_output.mp4)
* Link [Harder Challenge Video Output](./harder_challenge_video.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had some issues with the thresholding discussed in step 2. I first tried to use the "and" operator for combining the outputs, but the result was very "weak" lane lines in the binary image, since the gradient thresholding did not produce "solid" areas of white were the lane lines where. When I switched to the "or" operators, the gradient thresholds introduced too much noise. However, the L-channel threshold did not pick up on yellow lines well only the white. The S-channel did a good job with yellow lines, but freaked out (aka noise-apocalypse) when there shadows in the image. I googled around abit, and read the discussion forums and ended up using the B-channel from the LAB colorspace, which worked well. 

I have done some reflection on where my pipeline could fail:
* It fails on both challenge videos. This might be due to hardcoded src and dst points, as there is some elivation, at least in the "hardest challenge video". 
* It would be interesting to try this on an image with alot of snow (lots of white color). I'm guessing the result would be noisy, to put it mildly, and my sliding window techniques with not like it one bit. 
* Lots of white cars in the image, not sure what would happen though. 


I have done some reflection on what could improve my pipeline wrt. to robustness:
* Dynamic thresholding: Different thresholds for different sections of the image, thresholds based on number of pixels found etc. 
* Much more robust sanity checks on the fits: My current sanity checks are fairly basic. I could calculate some kind of "confidence" metric for each of the two fits, and reject both of the confidence in one deviates too much from the other. The idea here is to "force" only paralell fits. 
 





