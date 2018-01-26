#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[image1m]: ./my-images/explore_RGB.png "Random RGB images from X_train"
[image2m]: ./my-images/hist.png "Histogram of classes"
[image3m]: ./my-images/grayscale.png "Random grayscale images from X_train"

[image4m]: ./found-traffic-signs/4r.png 
[image5m]: ./found-traffic-signs/12r.png 
[image6m]: ./found-traffic-signs/13r.png 
[image7m]: ./found-traffic-signs/14r.png 
[image8m]: ./found-traffic-signs/17r.png 
[image9m]: ./found-traffic-signs/33r.png 

[image10m]: ./my-images/softmax.png 

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

## Submission Files
All files in the original project are included, excluding the "traffic-signs-data" folder, according to the project description. 

## Dataset Summary
Size, number of examples (for training, testing and validation) and number of classes are included.

## Exploratory Visualization
The submission shows both RGB images, and images after preprocessing. Label distribution is visualized through a histogram. 

## Preprocessing 
Preprocessing is described and discussed.

## Model Architecture
The model architecture is explained, and discussed. Comments in code provides information about each layer. 

## Model Training
Model training is described and discussed. 

## Solution Approach
Final accuracy was 0.956 on the validation set. 

## Acquiring New Images
Submission includes six new german traffic signs found on the web. 

## Performance on New Images
Accuracy on both the test set from Udacity and the traffic signs I found are calculated. Preformance comparisson is discussed. 

## Model Certainty - Softmax Probabilities
The top five softmax probabilities are outputted, and their uncertianty is discussed. 

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/olasson/CarND-Self-Driving-Car/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32, 32, 3)
* The number of unique classes/labels in the data set is: 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is random RGB and images, and a historgram of the traffic sign classes.

![alt text][image1m]
![alt text][image2m]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I converted to grayscale mainly because it speeds up training. 

Here is an example of traffic signs after grayscaling.

![alt text][image3m]

I normalized the data because the it also speeds up training, and reduces the chance of getting stuck in a local optima.  


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| Input |Output| 
|:---------------------:|:---------------------------------------------:| :----:|:-----:|
| Convolution 5x5     	| 1x1 stride, valid padding, RELU activation 	|**32x32x1**|28x28x6|
| Max pooling			| 2x2 stride, 2x2 window						|28x28x6|14x14x6|
| Convolution 5x5 	    | 1x1 stride, valid padding, RELU activation 	|14x14x6|10x10x16|
| Max pooling			| 2x2 stride, 2x2 window	   					|10x10x16|5x5x16|
| Convolution 5x5 		| 1x1 stride, valid padding, RELU activation    |5x5x16|1x1x400|
| Flatten				| Reduce layer 2 to 1 dimension					|5x5x16| 400|
| Flatten				| Reduce layer 3 to 1 dimension					|1x1x400| 400|
| Concat				| Combine layer 2 and 3					|400 + 400| 800|
| Dropout				| Dropout on combined layer					|800| 800|
| Fully Connected | Output = number of traffic sign classes			|800|**43**|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

## Optimizer
I used the Adam optimizer. It is very popular in deep learning, and I use it as the "go to" optimizer. It uses Adaptive Gradient Algorithm (AdaGrad) which maintains a per-parameter learning rate that improves performance on problems where sparse gradients is an issue (computer vision problems). Additionally, it uses Root Mean Square Propagation (RMSProp) which also maintains per-parameter learning rates and helps when dealing with noisy data. 

## Hyperparameter Summary & Discussion
* EPOCHS = 65
* BATCH_SIZE = 100
* SIGMA = 0.1
* MU = 0
* LEARNING_RATE = 0.0009
* ACCURACY_THRESHOLD = 0.960


I mainly tuned the learning rate. For very low learning rates I got horrible results, I'm guessing that the optimizer had trouble with convergence, but I'm not sure. 

After some very useful Udacity feedback, I added the ACCURACY_THRESHOLD parameter. If the validation accuracy is greater than or equal to this, the training stops.
* Pros: Avoids the scenarios I ran into before, where EPOCH 65 produced worse accuracy than EPOCH 10. 
* Pros: If tuned sensibly, can help countering overfitting. 
* Cons: Effectively caps the accuracy of the model at ACCURACY_THRESHOLD. My model stopped at EPOCH 11, but I don't have any guarantees that any later EPOCH would not produce better accuracy. This however, can be subjected to the trial and error method, to see if a better accuracy can be achieved. Time constraints prevents me from doing this right now. 
* Overall a very simple, straightforward way of deciding when to stop training. In my case, I achived both better validation accuracy and faster training time with this new parameter.    

## 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code is found in the 9th cell of the Ipython notebook.

## My final model results were:
* training set accuracy of: I have to admit, I did not calculate this. 
* validation set accuracy of 0.961
* test set accuracy of 0.940
* test set (images I found) accuracy of 1.000

## If a well known architecture was chosen:
* What architecture was chosen?
  * I chose to follow the architecture in http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf. 
* Why did you believe it would be relevant to the traffic sign application?
  * It concats layer 2 and 3 to provide the classifier with different scales of the receptive fields. I thought this would be useful since my preprocessing was so simple (no scaling, translation etc.). 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 * I suppose the best argument I can provide is the accuracy on the images I found. I'm not sure what to conclude exactly based on the udacity validation and test set results. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4m] ![alt text][image5m] ![alt text][image6m] 
![alt text][image7m] ![alt text][image8m] ![alt text][image9m]

Honestly, none of the images should be too difficult to classify. All of them are clearly visible. It would have been more interesting with a more difficult test set with partially broken/ obscured images etc. I'll keep that in mind for the future. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/h     		| 70 km/h   									| 
| Priority road    			| Priority road 										|
| Yield					| Yield											|
| Stop	      		| Stop					 				|
| Turn right ahead			| Turn right ahead      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.953. As mentioned, my test set was not that difficult, so I suppose the increase in accuracy is not that suprising. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for the softmax probabilities is found in the 14th cell of the Ipython notebook.


![alt text][image10m]

As the image shows, the model is 100% certain of all images. It would seem that my model is fairly reliable when it comes to predicting traffic signs, although I acknowledge that a much harder test set would probably not lead to this kind of accuracy. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


