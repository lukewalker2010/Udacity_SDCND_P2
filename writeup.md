#**Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report (this document)


[//]: # (Image References)

[image1]: ./Writeup_Images/5Random.png "5 Random Images"
[image2]: ./Writeup_Images/Initial_Histogram.png "Starting Histogram"
[image3]: ./Writeup_Images/Greyscale.png "Grayscale"
[image4]: ./Writeup_Images/Final_Histogram.png "Final Histogram"
[image5]: ./Internet_Images/1.jpg "Traffic Sign 1"
[image6]: ./Internet_Images/2.jpg "Traffic Sign 2"
[image7]: ./Internet_Images/3.jpg "Traffic Sign 3"
[image8]: ./Internet_Images/4.jpg "Traffic Sign 4"
[image9]: ./Internet_Images/5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799  which is  67.12899554389551 % of total images.
* The size of the validation set is 4410  which is  8.50710854761859 % of total images.
* The size of test set is 12630  which is  24.363895908485887 % of total images.
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First I plotted 5 random images to get a feel of the dataset.

![alt text][image1]

Next, to get a better idea of the dataset I plotted a histogram of the classes.

![alt text][image2]

As you can see right away some classes have way more images than others.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

Next, I normalized the image data so that the network would behave properly.

Finally, I decided to generate additional data because of the initial histogram showed that some classes were lacking images.

To add more data to the the data set, I rotated the images a random angle out of the following set [-25, 25, -20, 20, -15, 15, -10, 10, -5, 5]

After augmenting the data the histogram of the data set looks like:

![alt text][image4]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x1 Grayscale image   						      	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6   	|
| RELU					        | Activation											              |
| Average pooling	      | 2x2 stride,  outputs 14x14x6 			          	|
| Convolution 5x5	      | 1x2 stride, valid padding, outputs 10x10x16 	|
| RELU	                | Activation                                    |
| Average pooling	      | 2x2 stride,  outputs 5x5x16 			          	|
| Flatten       	      | outputs 400                 			          	|
| Fully Connected	      | outputs 200                 			          	|
| Dropout       	      | keep probability 0.75       			          	|
| Fully Connected	      | outputs 100                  			          	|
| Dropout       	      | keep probability 0.75       			          	|
| Fully Connected	      | outputs 43 			                            	|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Epoch of 10 and batch size of 150. I stuck with a mu of 0 and sigma of 0.1. I used a learning rate of 0.0025.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.985
* validation set accuracy of 0.944
* test set accuracy of 0.913

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I started with the lenet architecture, because I thought that the traffic signs were similar to the MNIST images in size and function. and then switched from max pooling to average pooling. I also added in droputout to try to prevent overfitting of the model. Based on my results I believe that this model is a good fit for the data, but some improvements could be made as discussed in the next section.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9]

The last image may be difficult to classify because it is a bicycle sign in a circle and the training bicycle images were on triangle shaped signs.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (50km/h) 	| Speed limit (50km/h)        						      |
| Bumpy road      			| Bicycles (2nd Prediction was Bumpy Road) 	  	|
| Road work					    | Road work               											|
| General caution    		| General caution             					 				|
| Bicycles         			| Speed limit (50km/h)            							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of 0.944

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 73rd cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .17               		| Speed limit (50km/h)   									|
| .10     		      		| Speed limit (30km/h) 										|
| .03					          | Speed limit (70km/h)										|
| .02	      		      	| Speed limit (100km/h)					 				|
| .001				          | Speed limit (80km/h)      							|

For the second image the second prediction was correct and only 1% below the first prediction

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .16               		| Bicycles crossing   									|
| .15     		      		| Bumpy road 										|
| .03					          | No vehicles										|
| .02	      		      	| Turn left ahead					 				|
| .005				          | Children crossing     							|

For the third image the first prediction was correct and had a much higher probability than any of the following predictions

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .23               		| Road work   									|
| .06     		      		| Wild animals crossing 										|
| .05					          | Double curve										|
| .02	      		      	| Road narrows on the right					 				|
| .003				          | General caution      							|

For the fourth image the first prediction was correct and had a much higher probability than any of the following predictions.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .41               		| General caution   									|
| .25     		      		| Road work 										|
| .08					          | Pedestrians									|
| .03	      		      	| Right-of-way at the next intersection 				|
| .02 				          | Speed limit (20km/h)      							|

For the fifth image all the probabilities were low and none were correct. I suspect this is from the shape of the sign and could be fixed by augmenting the data with more signs.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .09               		| Speed limit (50km/h)   									|
| .02     		      		| Wild animals crossing 										|
| .004				          | Speed limit (100km/h)									|
| .005      		      	| Double curve					 				|
| .006				          | Go straight or left     							|

Overall I enjoyed this project and thought it was very interesting. I look forward to using what I learned in future projects.
