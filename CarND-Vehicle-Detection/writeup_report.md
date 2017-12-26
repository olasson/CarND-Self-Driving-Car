## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./my-images/random_car_imgs.png
[image2]: ./my-images/random_noncar_imgs.png
[image3]: ./my-images/car_hog_features.png
[image4]: ./my-images/noncar_hog_features.png
[image5]: ./my-images/false_positives.png
[image6]: ./my-images/scale_1_5.png
[image7]: ./my-images/search_space.png
[image8]: ./my-images/search_multiple.png
[image9]: ./my-images/hmap_before_thresh.png
[image10]: ./my-images/hmap_after_thresh.png
[image11]: ./my-images/pipeline_output.png
[image12]: ./my-images/pipeline1.png
[image13]: ./my-images/pipeline2.png
[image14]: ./my-images/pipeline3.png


[video1]: ./test_video_output.mp4
[video2]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In cell 7, I loaded the `vehicle` and `non-vehicle` images using `glob`. Next, I took a look at a random set of 25 car images and 25 non-car images using the helper function `show_random_imgs()` (found in code cell 2). The visualization can be seen below

![alt text][image1]

![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed one image from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. The hog feature extraction is done with the function `get_hog_features`. The function implementation is found in code cell 3, and the hog features are extracted in code cell 10. Here is the output using my final choice of paramters (I'll discuss briefly how I ended up with those in the next section) 


![alt text][image3]
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

These parameters were found primarily by trial and error. The metric I used for evaluating any given set of parameters was the accuracy produced by the SVM. I first extracted the features using the function `extract_features()` found in code cell 3. I went through each "available" color space. I made some observations


* The setting `hog_channel` = 'ALL', had a noticeably negative impact on the extraction time, but tended to yield >97% test set accuracy for pretty much all color spaces. 
* For some reason, `RGB` tended to produce worse training times. 
* The colorspace `YUV` tended to produce very quick traning times. 
* Increasing `pix_per_cell` tended to speed up training and extraction times without any major negative impact on accuracy

In the end it came down to

* `colorspace` = 'HSV'
* `orient` = 9
* `pix_per_cell` = 8
* `cell_per_block` = 2
* `hog_channel` = 'ALL'

VS.

* `colorspace` = 'YUV'
* `orient` = 11
* `pix_per_cell` = 16
* `cell_per_block` = 2
* `hog_channel` = 'ALL'

The `HSV` set had slightly better accuracy than the `YUV` set, but the `YUV` set "won" since it had a much better training and (perhaps more importantly) extraction time. My final choice of paramters is found in code cell 10. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Extracting the features for training happens in code cell 11. The training itself happens in code cell 12. Note that I **ONLY** used the HoG features, not spatial intensity or channel intensity. The Linear SVM classifier was trained using the default parameters of `sklearn`. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My sliding window function is called `find_car_rectangles()` and is found in code cell 4. The code is based on the Udacity function `find_cars()`. The function combines extracted HOG features with a sliding window search. Instead of performing feature extraction on each window individually (which can be computationally expensive), the HOG features are extracted for the entire image (or a selected portion of it) defined by `y_start` and `y_stop`. Next, these features are subsampled according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a "this is a car" prediction. Instead of overlap `find_car_rectangles()`, defines how many cells to step in te x and y direction. I added an optional paramter called `vis` which forces the function to return every rectangle it tried for visualization purposes. 


Here are two examples of `find_car_rectangles()` applied to the image `test1.jpg` with scale 0.5 and 1.5 respectively

![alt text][image5]

![alt text][image6]

As one can see, scale = 0.5 produced alot of false positives, and I decided to only use scales greater or equal to 1.0. I used `vis` to visualize the search space. An example with scale = 1.5, can be seen below

![alt text][image7]


Next, `find_car_rectangles()` is used by the function `search_multiple_scales_and_areas()` (found n code cell 6) which sets up a pre-defined set of `scales` and `y_start`, `y_stop` values. It produced the following output

![alt text][image8]

A heatmap is used for filtering out false positives. First, the funtion `add_heat()` (found in code cell 6) is used to add "heat" for all pixels inside each box. The overlapping areas of multiple boxes (see previous image) will then "glow" the brightest. An example output of `add_heat()` is seen below

![alt text][image9]

The function `apply_threshold()` (found in code cell 6) was then used to only select areas that "glowed" brighter than a given threshold (default value of 1 was used in this example)

![alt text][image10]

The `scipy.ndimage.measurements.label()` function was used to indentify spatially contiguous areas of the thresholded heatmap and assigns each a label. These labels were then fed to `draw_labeled_rectangles()` which draws the rectangles. The final result of this "pipeline" is seen below

![alt text][image11]



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are three examples of applying the pipeline to test images (from code cell 18).

![alt text][image12]

![alt text][image13]

![alt text][image14]

The final implementation performs decently. It is able to identify the nearby vehicles in each of the frames with one false positive for the entire video.

I have included some old attempts at the project and test video in the folder `old_attempts`. The biggest problem was false positives, which I fixed by tuning the heatmap threshold. The biggest optimization of the pipeline was changing the `pixels_per_cell` parameter from its default value to 16. It was a non-trivial speed up in terms of computation time, at the cost of a small loss of accuracy. 

The other main optimization was changing the threshold. I tried constant values but they did not work very well. For example, high constant values tended to underestimate the size of the vehicle. I ended up using a dynamic threshold, which I will explain in 
**Video Implementation, section 2**. 


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

* Here's a [link to my video project video result](./project_video_output.mp4)
* Here's a [link to my video test video result](./test_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The function `pipeline()` is found in code cell 6. I added an option `use_prev` which switches between an "advanced" and "naive" version of the pipeline. The naive version was only used for testing, and the default is the advanced version. In the advanced version, a class called `Rectangles_Data()` (see bottom of code cell 6) is used to store some number of previous rectangles. I got this idea from the `Line()` class from the preivous project. Instead of preforming thresholding for one one frame at the time, the previous rectangles are fed to the `add_heat()` function and thresholded with  `4 + len(rectangle_data.prev_rectangles)//3` which was found by trial and error. The idea was to create a "moving-average" type filter that rejects false positives. 


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems encountered was mainly concerned with detection accuracy. Finding a balance between speed and accuracy was difficult. 



**Where my pipeline would probably fail:** 
* If the pipeline encounters a vehicle that produces HoG features it is unable to cope with (perhaps some old-school car would confuse it). 
* Some background-to-vehicle color schemes such as white car against white background might confuse the classifier. 
* My pipeline is just flat out terrible at detection oncoming traffic and far-away cars. 


**Possible improvements:** 
* Improve the accuracy
* Predict vehicle location based on previous frames and use it to limit search area. 
* A neural network could omitt the sliding window technique. 
* For oncoming traffic and far-away cars, I tried smaller scales, but it resulted alot of false positives.

