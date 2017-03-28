# **Traffic Sign Recognition** 
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

[image1]: ./examples/num_training_images_barchart.png "Number of Training Images by type"
[image2]: ./examples/normalized.jpg "normalized images"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./data/sign1.jpg "Traffic Sign 1"
[image5]: ./data/sign2.jpg "Traffic Sign 2"
[image6]: ./data/sign3.jpg "Traffic Sign 3"
[image7]: ./data/sign4.jpg "Traffic Sign 4"
[image8]: ./data/sign5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jrad77/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic signs data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  In it I show a sample of each type
of traffic sign and then plot the frequency of each sign in the training data set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because color is not a huge differentiator between the different sign types and it was useful for shrinking the input size. I then ran an adaptive histogram equalization step to enhance the contrast in the image as well as brighten up some of the darker areas and lower the intensity of some of the brighter areas in the images.

I then centered each image around its mean value. This was done to help the Stochastic Gradient Descent navigate its error surface.

Here are samples of each class normalized:
![alt text][image2]

#### 2. Provide details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer.

The model is laid out in fifth, sixth, and seventh code cells.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 30x30x16 	|
| tanh					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x16 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 13x13x16    |
| tanh					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x32 			     	|
| Convolution 1x1	    | 1x1 stride, same padding, outputs 6x6x32      |
| Fully connected		| 1152 input, outputs 240						|
| tanh                  |                                               |
| Fully connected       | 240 input, outputs 100                        |
| tanh                  |                                               |
| Fully connected       | 100 input, outpus 43                          |
| Softmax				| Softmax on the logits to output probabilities |
| Cross Entropy         |    Cross entropy to select prediction         |

I started with the basic LeNet from the previous lab and was able to get around
93% accuracy on the validation set. However, I skimmed through [the suggested paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and somehwat replicated its architecture by adding the extra convolutional layer and fully connected layers, as well as using tanh instead of relus. Using tanh instead of the relus actually resulted in more than 1% gain in validation accuracy.

I tried to ensure that the model would be 'too big' for the data sets and went as far as to add a 1x1 convolutional layer to add parameters without transforming the image.

Using this model I was able to acheive over 96% on validation accuracy and 
95% accuracy on the test set, indicating that the model was not overtrained.

#### 3. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh, eigth, and ninth cells of the ipython notebook. 

To train the model, I used the Adam Optimizer which computes adaptive learning rates for each parameter and utilizes 'momentum' to help steer the gradient descent towards the minima. The function being minimized was the simple cross entropy of the one-hot encoded labels.

I used a batch size of 128, a learning rate of 0.0001, and 100 epochs. I played with larger batch sizes, but found I needed to increase the number of epochs to make up for the fewer iterations of gradient descent.

#### 4. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eigth and ninth cells of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.965 
* test set accuracy of 0.950

The first architecture I tried was LeNet from the previous lab. With almost no modifications I was able to get around .92-.93 on the validation set. This network did not seem to be deep enough, and scaled the feature map width and heights down very quickly, possibly losing information in the process.

I then skimmed [the suggested paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and decided to make my network deeper and use tanh instead of relu for my non-linearities. I then started
to get higher accuracy on the validation set, closer to 0.94. I then implemented the code for viewing
the feature maps at different points in the network. This lead me to use smaller convolutional filters in the first two stages and keep the width and heights on the feature maps larger than LeNet to keep more information around until deeper in the network.

I did not run the included test set until I had settled on a model. 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the twelth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Pedestrians      		| Pedestrians   									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection 										|
| Road work					| Road work											|
| No vehicles	      		| No vehicles					 				|
| Speed limit (30km/h)			| Speed limit (30km/h)      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook. The softmax values are graphed in cell 15.

For the first image, the model is relatively sure that this is a Pedestrians sign (probability of 0.87), and is correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .87         			| Pedestrians  									| 
| .12     				| Traffic signals 										|
| < .00					| Right-of-way at the next intersection											|
| < .00	      			| Speed limit (100km/h)					 				|
| < .00				    | Children crossing     							|


For the second and third image, the model was very certain with probability of .99 for both. It correctly predicted that the second image was the Right-of-way at the next intersection and the third as  Road work.

For the fourth image, the model was very certain of the sign being 'No Vehicles' (probability 0.95). It was correct. Top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| No vehicles  									| 
| .03     				| No passing 					|
| < .00					| Keep right											|
| < .00	      			| Speed limit (50km/h)					 				|
| < .00				    | Turn right ahead      							|

Finally, for the fifth image, the model was relatively certain that it is a 30km/h sign and was correct (probability of .84). Top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Speed limit (30km/h)  									| 
| .03     				| Stop 					|
| < .00					| Priority road											|
| < .00	      			| General caution					 				|
| < .00				    | Wild animals crossing      							|



