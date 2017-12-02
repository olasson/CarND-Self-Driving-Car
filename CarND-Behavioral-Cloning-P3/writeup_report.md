#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

[img1]: ./images/before_pre_processing.png "Image before pre-processing"
[img2]: ./images/after_pre_processing.png "Image after pre-processing"
[img3]: ./images/rotated_img.png "Image with random rotation applied"
[img4]: ./images/shifted_img.png "Image with random shift applied"
[img5]: ./images/steering_angle_before_flattening.png "Steering angle distribution before flattening"
[img6]: ./images/steering_angle_after_distribution.png "Steering angle distribution before flattening"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on [this](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper. This nVidia networch architecture was suggested by Udacity. 


####2. Attempts to reduce overfitting in the model

After reading through the Udacity discussion forms, I decided to not include Dropout. Instead I attempted to reuduce overfitting through other measures:
* Flatten the steering angle distribution to avoid HUGHE amounts of "0-angle" inputs. (see model.py line 47)
* Paying close attention to the change in loss during training. For example, with 5 epochs I noticed the loss stagnated completely at epoch 4. The resulting model was a terrible driver. I reduced the number of epochs to counter this. 

To create the validation set, I sectioned off 20% of the training data and applied a random shift and rotation to each image.  

####3. Model parameter tuning

The model used an adam optimizer, but I did set the learning rate to 0.0001 instead of the defaul 0.001, based on the Udacity discussion forums (model.py line 273).

####4. Appropriate training data

Key points in my data collection/selection strategy:
* Use all three cameras
* Collect alot of data, driving both clockwise and counter-clockwise.
* Use histograms to visualize steering angles, and to decide what data to throw away.
* Record specific "scenarios" where the car is "recovering" (car drives toward the track egde, but manages to get back to the center)
* Augment the right and left images with a steering angle offset to help "simulate" recovery (mentioned in Project: Behavioral Cloning: "Using Multiple Cameras"). As far as I can tell, it also helps with simulating a "recovery" scenario. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to look at what network architectures already existed for this kind of problem. The choice fell on the nVidia network architecture that Udacity mentioned. I thought this model might be appropriate because it has already proven that it is capable of solving this kind of problem (it has been used on real cars). The model (as far as I can tell from the paper) does not explicitly mention any activation function, so I chose my "default", the almighty ReLU. ReLU helps counter the vanishing gradient problem, and speeds up learning. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Additionally, to make it even harder for the model, I applied a random shift and rotation to the validation data. I have included some old models I trained, the main result from these models were changes to parameters, not the architecture itself. 

####2. Final Model Architecture

The final model architecture (model.py lines 124 - 143) consisted of a convolution neural network summarized in the following table:

```
Layer                 Function                                   Output               
=======================================================================================
(Lambda)              Normalization                              (None, 66, 200, 3)   
_______________________________________________________________________________________
(Convolution2D)       5x5 Kernel + relu                          (None, 31, 98, 24)   
_______________________________________________________________________________________
(Convolution2D)       5x5 Kernel + relu                          (None, 14, 47, 36)    
_______________________________________________________________________________________
(Convolution2D)       5x5 Kernel + relu                          (None, 5, 22, 48)     
_______________________________________________________________________________________
(Convolution2D)       3x3 Kernel + relu                          (None, 3, 20, 64)     
_______________________________________________________________________________________
(Convolution2D)       3x3 Kernel + relu                          (None, 1, 18, 64)     
_______________________________________________________________________________________
(Flatten)             Make feature vector for classifier         (None, 1152)          
_______________________________________________________________________________________
(Dense)               Classifier                                 (None, 100)           
_______________________________________________________________________________________
(Dense)               Classifier                                 (None, 50)            
_______________________________________________________________________________________
(Dense)               Classifier                                 (None, 10)            
_______________________________________________________________________________________
(Dense)               Classifier                                 (None, 1)             
=======================================================================================
```

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps: two clockwise, two counter-clockwise. A image from the center camera looks like this:

![alt text][img1]

Each image underwent 3 simple pre-processing steps: Cropping, to only include sections of the image that is usefull to the model (trees and water are not really that helpful). Resizing it to fit the model architecture, and finally shifting the color space to YUV (recommended by the nVidia paper). A pre-processed image looks like this:

![alt text][img2]

The next step was to recover data to teach the model how to "recover" if it veered off to the sides of the track. I did this for several sections of the track: once for the "yellow" striped edge, one for the dirt edge, once for the thick striped edge, and once for the bridge edge texture. I ended up doing the same for track two, but I only used one lap of training data for this track + some recovery scenarios. 

Based on the nVidia paper, I included two methods for augmenting the data: shift (model.py line 87) and rotation (model.py line 109). I used these methods to further augment my validation data, to make it harder for the model and hopefully get better generalization. 

Here is what a shifted image looks like: 

![alt text][img3]

Here is what a rotated image looks like:

![alt text][img4]

In the folder (old models), I've included some attempts. For model01, model02, I noticed that the car struggeled with the "recovery" scenario. I tried to go from 3 epochs to 5, which resulted in a horrible model (model03). This model clearly suffered from overfitting, (stagnation in loss, and increase in validation loss for last 2 epochs). I tried to go the other way, I lowered the number of epochs to 2, and made the data distribution flattening more aggressive. Additionally I increased the the steering angle offset (model.py line 182) to help with the recovery scenario. Model04, model05, model06, model07 are mainly the result of variation in how much i flattened the data distribution, and the steering angle offset. I was getting a little frustrated at this point. The main problem was that the car would come off the bridge too far to the left, and touch the edge. It also struggled with the left turn right after.I collected some more data, where I recorded this section one clockwise, and once counter clockwise. Finally, it worked, and the resulting model is the one I have submitted. 

My main takeaway from this project is: data collection and selection is incredibly important. You can have the best model architecture in the world, and it does not matter one bit if you're not sensible in how you collect and select your data. An important function in this project was flatten data distribution (model.py line 47). I used this as a tool to select what data to actually use for training. 

This is what the steering angle distribution would look like before flattening:

![alt text][img5]

This is what it would look like after flattening:

![alt text][img6]

The data generator (model.py line 1) uses the yield keyword to avoid loading all the trainig data into memory. Additonally, one can specific if it should augment the images by rotation and shift. For all images, it flips it aswell to increase the amount of data and help counter bias to one driving direction (the car pulled to the left for some of my very earliest attempts). 

####3. Some closing thoughts and future work

This was a really cool project. I wasn't able to make a generalized model that could drive both tracks, but I fully intend to revisit this project when time permits, with the specific goal of making a single model to drive both tracks. 

