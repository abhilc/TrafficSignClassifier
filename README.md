# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model arachitecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1.This is a README document of the steps taken in detail to accomplish traffic sign recognition.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. The Data was downloaded from the provided training, test and validation data set pickle files by Udacity. I used the python len function to just capture the length of these data sets

Summary statistics of the data sets are as follows

Number of training examples = 34799
Number of testing examples = 12630
Number of validation examples = 4410
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart where X-axis represents 43 unique classes and Y-axis represents the count of images in each class

<img src="https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/images_for_writeup/Screenshot%202021-06-26%20at%202.06.37%20PM.png"
     alt="BarChart"
     style="float: left; margin-right: 10px;" />

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because a single channel is sufficient. All that matters to the Covnet is the shape and size of the image regardless of the image being colored or not. 

After grayscaling, I also normalized this image using the formula (pixel - 128)/128. This will result in pixels having values between -1 and +1. This is sufficient for us because these values indicate the intensities of pixels. The below image shows the colored image, and the grayscaled normalized image below it

<img src="https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/images_for_writeup/before_after.png"
     alt="BarChart"
     style="float: left; margin-right: 10px;" />



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64                 |
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Training the model

To train the model, I used the following Hyperparameters

EPOCHS 100
BATCH_SIZE 128


My final model results were:
* training set accuracy of 99.3%
* validation set accuracy of 93.2%
* test set accuracy of 92.7%

<img src="https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/images_for_writeup/validation_accuracy.png"
     alt="BarChart"
     style="float: left; margin-right: 10px;" />

An iterative approach was chosen 
* The architecture of the model is a well known architecture initially proposed by Yann Lecun called LeNet. A schematic of the architecture is as below

<img src="https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/images_for_writeup/1_1TI1aGBZ4dybR6__DI9dzA.png"
     alt="BarChart"
     style="float: left; margin-right: 10px;" />

* Although the architecture is well known, I had to tune the hyperparameters manually to and run the training set several times so that the desired accuracy is achieved
* Underfitting: With only 50 EPOCHS and no preprocessing of the training images, the overall training accuracy achieved was around 80% for training set and 60% for validation set
* Then, using the numpy library the images were converted to grayscale. The shape of the images after conversion were 32x32x1. The 3 channel RGB was converted to a single channel
* I also normalized the images using (pixel - 128)/128. This converted the pixel values to range between -1 and 1. These could be interpreted as intensities of each pixel
* After the mentioned improvements the validation set accuracy was improved to around 91% which was still not enough to satisfy the project requirements
* As last steps I only increased the EPOCHS to 100, and plotted the validation accuracy against EPOCHS. The average accuracy across all batches was improved to 93.2%
* As the training set, test set and validation set accuracies are closer to each other, we can say that the model behavior is not random and it has trained well. 
 

### Testing the Model on new images 

#### 1. I downloaded 5 images from the web. 

Here are five German traffic signs that I found on the web:

![alt text][../downloaded_images/11_rigtoffway_atnextintersection_32x32x3.jpg] 
![alt text][../downloaded_images/12_priority_road_32x32x3.jpg] 
![alt text][../downloaded_images/17_noentry_32x32x3.jpg] 
![alt text][../31_wildanimalscrossing_32x32x3.jpg] 
![alt text][../34_turn_left_ahead.jpg]

All the images were classified correctly

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right_off_way    		| Right_off_way									| 
| Priority     			| Priority 										|
| No Entry				| No Entry										|
| Wild Animal	   		| Wild Animal					 				|
| Left Ahead			| Left Ahead        							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100. This compares favorably to the accuracy on the test set of 92%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the top 5 softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|





