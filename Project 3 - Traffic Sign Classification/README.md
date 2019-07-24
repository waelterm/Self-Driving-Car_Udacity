# **Traffic Sign Recognition** 

## Writeup

### This is the documentation of Project three of the Udacity Self-Driving Car Nanodegree.

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

[image1]: ./explore_dataset/label_distribution_test.png
[image9]: ./explore_dataset/label_distribution_train.png "Visualization Taining Set"
[image10]: ./explore_dataset/label_distribution_valid.png "Visualization Validation Set"
[image11]: ./explore_dataset/test.jpg "Sample Image Test Set"
[image12]: ./explore_dataset/train.jpg "Sample Image Taining Set"
[image13]: ./explore_dataset/valid.jpg "Sample Image Validation Set"


[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./my_test_images/Formatted/sign1.png "Traffic Sign 1"
[image5]: ./my_test_images/Formatted/sign3.png "Traffic Sign 2"
[image6]: ./my_test_images/Formatted/sign4.png "Traffic Sign 3"
[image7]: ./my_test_images/Formatted/sign6.png "Traffic Sign 4"
[image8]: ./my_test_images/Formatted/sign7.png "Traffic Sign 5"

[image14]: ./my_test_images/Formatted/sign9.png "Traffic Sign 6"
[image15]: ./my_test_images/Formatted/sign11.png "Traffic Sign 7"
[image16]: ./my_test_images/Formatted/sign10.png "Traffic Sign 8"


[image18]: ./TrainingAnalysis/LR0_001.png "Learning Rate 0.001"
[image19]: ./TrainingAnalysis/LR0_005.png "Learning Rate 0.005"
[image20]: ./TrainingAnalysis/LR0_0005.png "Learning Rate 0.0005"
[image21]: ./TrainingAnalysis/BatchSizes.png "Batch Sizes"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! The code associated with this project is in this [Github](https://github.com/waelterm/Self-Driving-Car_Udacity/tree/master/Project%203%20-%20Traffic%20Sign%20Classification)
You will also find the output of the evaluations of the final model in the code_output.txt file.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used pandas to analyze the signnames.csv file. 
However, I wrote a custom script to analyzed the training, test and validation dataset. 
This script can be found in the analyze_dataset.py file. These were the obtained results:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

From plotting the number of occurences in each dataset, it is obvious that the occurences of each type of sign vary largely. This might have an effect on the performence of the network. It could be that the network performs better on the more frequent signs in the dataset.
However, this effect will not be seen in the test or validation set because all datasets have the same distribution of occurences.
##### Distribution of Test Set
![alt text][image1]
##### Distribution of Training Set
![alt text][image9]
##### Distribution of Validation Set
![alt text][image10]

The following are some sample images with labels from each dataset. #WORK IMAGES ARE DISPLAYED WRONG 
##### Sample Image from test set (Speed limit (50km/h))
![alt text][image11]
##### Sample Image from training set (End of no passing)
![alt text][image12]
##### Sample Image from validation set (Go straight or right)
![alt text][image13]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided against using greyscale images after considering that the color information is very valuable when trying to identify traffic signs.

The only steps that has been taken to preprocess the images is normalization. The uniform distribution makes sure that no very large numbers dominate the outcome of the network. We want the network to learn from all of the pixels in the image not only from some very bright spots. Normalizing also makes the network converge faster leading with better training results.

Future improvements could be: Zoom in and rotate images to create additional images.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout				| KP: 0.9												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16       									|
| RELU					|												|
| Dropout				| KP: 0.9												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten       		| output 400        									|
| Fully Connected		| output 200        									|
| RELU					|												|
| Dropout				| KP: 0.9												|
| Fully Connected		| output 84        									|
| RELU					|												|
| Dropout				| KP: 0.9												|
| Fully Connected		| output 43        									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained several models to identify how the hyperparameters would influence the networks performance:
Each training period lasted 100 epochs, and was evaluated after each epoch. The adam optimizer has been used during each of the stages in the training.

During the first training set, I varied the learning rate and the keeping probability of the dropout layers:

    LEARNING_RATES = [0.005, 0.0005, 0.001]
    KEEP_PROBS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    BATCH_SIZES = [128]
    
The detailed data can be found in the "training_summary.xlsx" file.
The following plots show some results of the different hyperparameter configurations:

![alt text][image18]

![alt text][image19]

![alt text][image20]


This list shows the maximum achieved accuracy during all epochs for each configuration:




|Parameter Description	|MAX
|:---------------------:|:---------------------------------------------:| 
|LR: 0.0005 KP: 0.5	|0.877551021
|LR: 0.0005 KP: 0.6	|0.913378685
|LR: 0.0005 KP: 0.7	|0.930839002
|LR: 0.0005 KP: 0.8	|0.931519274
|LR: 0.0005 KP: 0.9	|0.943764173
|LR: 0.0005 KP: 0.95|	0.946031746
|LR: 0.0005 KP: 1	|0.930839002
|	|
|LR: 0.005 KP: 0.5	|0.74829932
|LR: 0.005 KP: 0.6	|0.84739229
|LR: 0.005 KP: 0.7	|0.901814059
|LR: 0.005 KP: 0.8	|0.908390023
|LR: 0.005 KP: 0.9	|0.926984127
|LR: 0.005 KP: 0.95	|0.929705216
|LR: 0.005 KP: 1	|0.949206349
|	|
|LR: 0.001 KP: 0.5	|0.885034014
|LR: 0.001 KP: 0.6	|0.920408163
|LR: 0.001 KP: 0.7	|0.939002267
|LR: 0.001 KP: 0.8	|0.93537415
|*LR: 0.001 KP: 0.9*	|*0.951020408*
|LR: 0.001 KP: 0.95	|0.947165533
|LR: 0.001 KP: 1	|0.941723356

It can be seen that the network with a learning rate of 0.001 and a keeping probability of 0.9 performed best.
Therefore, I decided to use this preliminary network to investigate the effect of changing the batch size.

    LEARNING_RATES = [0.001]
    KEEP_PROBS = [0.9]
    BATCH_SIZES = [32, 64, 128, 256, 512]

From the table, it can be seen that changing the batch sizes does not have any relevant effects on the network performance.
        

![alt text][image21]


Due to the irrelevance of the batch size, the following Hyperparamters were choosen to train and save the network:

    EPOCHS = 69
    LEARNING_RATES = [0.001]
    KEEP_PROBS = [0.9]
    BATCH_SIZES = [128]

The network has been saved in the "saved_models" folder. The model has been tested with the test dataset.
The code for this can be found in the test_accuracy.py file. The test accuracy was found to be 94.51%.




#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.



My final model results were:
* training set accuracy of 99.93%
* validation set accuracy of 94.51% 
* test set accuracy of 94.01%

The large difference of the validation accuracy compared to the training accuracy indicates that the network is already overfitted. 
It might be wise to decrease the number of epochs in the future. Additionally, it might be a good idea to use a larger network, as well as a larger dataset.
The later can be realized by augmenting the existing data.

The process of finding the final model has been described in section 3 because most of the work has been done on the hyperparameter tuning.
The only major changes to the LeNet architecture have been:
* Adding Dropout to every layer to prevent overfitting.
* Adjusting the input to take RGB images. The color information is very helpful for humans when classifying traffic signs. Therefore, I thought that important information should not be omitted.
* Adjusting the output to work with 43 classes. This was a necessary step because there were 43 categories.
* Adjusting the number of nodes in the last hidden layer. The number of nodes in the output layer was increased due to the number of classes. 
Therefore, I adjusted the number of nodes in the last hidden layer for a more linear decrease in the number of nodes.


If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

With a large dataset, like the one used in this case, I was not sure how high the chances would be that a random traffic sign from the internet would come from the same dataset.
Therefore, I decided to ask some friends in Germany to take pictures of street signs they saw. The cropped results can be seen below.

Discussion of factors that might impact the classification:
**Image 1:** The stop sign is well lit and in the center of the image. However, there is a ripped off sticker at the bottom of the street sign. This augmentation of the sign might make it more difficult for the network to classify it.

**Image 2:** The yield sign is not blurry and provides a clear contrast to the background. However there is a wide poll behind it and a street light below. These additional features in proximity to the sign might make classification harder.

**Image 3:** The second yield sign is in the center of the image, but due to the lighting conditions, the outside of the sign seems brown rather than red. Additionally  part of a Left Turn Ahead sign can be seen. Both of these items might make it harder for the network to classify the image

**Image 4:** The fourth image shows a Trucks prohibited sign. This sign was not in the original dataset and I expect the network to misclassify it with a lower confidence. In addition to that, there are difficult lighting coditions due to uneven lighting accross the sign. Furthermore, the image is taken from below. This angle might make the classification more challenging than an image directly facing the sign. Lastly, the sign itself is partially washed off making it hard even for a human to imideately know what type of sign it is.

**Image5:** The Left Turn Ahead sign, is can clearly be seen and is in the center of the image. However a very busy background as well as part of a Yield sign above the Left Turn Ahead sign might make it challenging to classify the image.

**Image 6** The Right Turn Ahead sign is not blurred and can clearly be seen. owever, there is a sticker at on the street sign. This augmentation of the sign might make it more difficult for the network to classify it. Additionally, part of a stop sign, and part of a pedestrians only sign can be seen. These might further impact the network performance.

**Image 7** The speed limit (70 km/h) image is very bright with strong contrast. It looks almost unnatural. THe intense colors might provide a challenge for the network. Furthermore, there is a strong contrast between the sky and the trees, this might be picked up as a feature of the sign even though it is not.

Image 1

![alt text][image5] 

Stop Sign identified as Stop Sign

Image 2

![alt text][image6] 

Yield Sign identified as Yield Sign

Image 3

![alt text][image7] 

Yield Sign identified as Yield Sign

Image 4

![alt text][image8] 

Undefined Sign classified as Speed Limit (60 km/h)

Image 5

![alt text][image14] 

Turn left ahead Sign classified as turn left ahead Sign

Image 6

![alt text][image16]

Turn right ahead Sign classified as turn right ahead

Because those images did not include a single speed limit sign, I added an image I found on the web to the test images:

Image 7

![alt text][image15]


The code for showing the images and classifying them can be found in the "new_test_images.py" file.
The new test images have been cropped to have the same height as width and to center the sign to be classified. 
The cropped images can be found in the my_test_images/Formatted folder. The original images can be found in the 
my_test_images folder. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).



Here are the results of the prediction:

| Image			        |     Prediction	        					|  Confidence |
|:---------------------:|:---------------------------------------------:| :---------------------------------------------:| 
| Priority Road      		| Priority Road   									| 100%
| Turn right ahead    			| Turn right ahead 										| 99%
| Speed Limit (30 km/h)				| Speed Limit (70 km/h)											| 95%
| Stop	      		| Stop					 				| 99%
| Yield		| Yield      							| 100%
| Yield     | Yield |  88%
|Turn left ahead | Turn left ahead| 98%
| Unknown   | Speed Limit (60 km/h) | 99%


The model was able to correctly guess 6 of the 7 traffic signs that were within scope, which gives an accuracy of 86%. 
This does not hold up to the 94.5% accuracy of the test set. However, this deviation is not statistically relevant due to the small sample set.
The last test image is of a street sign that is out of scope (was not in the training dataset).
I was curious whether the network would show a decreased confidence and if it will detect similarities to the sign types it learned.
The guess of a speed limit sign definitely shows that the network generalizes because the structure of the sign is very similar to that of a speed limit sign. 
Yet, the confidence of the network is very high which is not expected for a sign type it has not seen.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
I choose to take a deeper look at the probabilities of the falsely classified Speed Limit Sign.
For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were



| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .946         			| Speed limit (30 km/h)   									| 
| .054     				| Keep right 										|
| .000001					| Speed limit (70 km/h)											|
| .0000005	      			| No passing					 				|
| .0000001			    | Speed limit (60 km/h)      							|

It can be seen that the network is very confident in its wrong prediction.
However, it seems to generalize well. All five predicted traffic signs are circular, Four of them have a red outer circle with a white interior, and three of them are speed limit signs. 

In general all of the predictions the confidence values were very high. This might be another indicated that the network is slight overfit.
To see the confidence values of the first five guesses for each of the test images, please refer to the code_output.txt file.
If you run the new_test_images.py file, it will show you each of the images and annotate them with the classification and confidence values.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
This will be a future improvement

###List of future improvements
* Add more layers
* Experiment with different optimizers
* Create more data using zoomed and rotated images
* Visualize neural network

