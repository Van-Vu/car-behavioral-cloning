# **Behavioral Cloning** 

[//]: # (Image References)

[image1]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/Center/left_2018_08_02_20_32_49_865.jpg "Center Driving"
[image2]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/Center/center_2018_08_02_20_32_49_865.jpg "Center Driving"
[image3]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/Center/right_2018_08_02_20_32_49_865.jpg "Center Driving"
[image4]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/Center_Reverse/left_2018_08_02_19_56_39_523.jpg "Center Driving"
[image5]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/Center_Reverse/center_2018_08_02_19_56_39_523.jpg "Center Driving"
[image6]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/Center_Reverse/right_2018_08_02_19_56_39_523.jpg "Center Driving"
[image7]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/Recovery/left_2018_08_02_20_12_49_759.jpg "Recovery Driving"
[image8]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/Recovery/center_2018_08_02_20_12_49_759.jpg "Recovery Driving"
[image9]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/Recovery/right_2018_08_02_20_12_49_759.jpg "Recovery Driving"
[image10]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/Bridge_Recovery/left_2018_08_04_13_09_54_220.jpg "Recovery Driving"
[image11]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/Bridge_Recovery/center_2018_08_04_13_09_54_220.jpg "Recovery Driving"
[image12]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/Bridge_Recovery/right_2018_08_04_13_09_54_220.jpg "Recovery Driving"
[image13]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/model.jpg "model"
[image14]: https://github.com/Van-Vu/car-behavioral-cloning/blob/master/Images/training_attemps.jpg "training attemps"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 (a video recording of my vehicle driving autonomously around the track for at least one full lap)
* this file writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

- My model consists of 5 convolution neural networks with 5x5, 3x3 filter sizes and depths between 24 and 64 (model.py lines 73-77). 

- The model also consists of 4 fully-connected layers depths 120 to 1 (moel.py lines 79-85)

- The data is normalized in the model using a Keras lambda layer (line 72). 

#### 2. Attempts to reduce overfitting in the model
- Each convolution layer has L2 regularizer to fight against overfitting (line 73 - 77)

- Each FC layer has L2 regulizer and dropout (rate 0.5) to fight against overfitting (lines 79 - 85)

- The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py code line 112-113). 

- The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

- The model used an adam optimizer with learning rate 0.0001 (model.py line 110).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of 4 datasets:
- Center lane driving
- Center lane driving Reverse (which has more right turns)
- Recovering from the left and right sides of the road
- Recovering when the car drive to the bridge and drive out of the bridge

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make sure that the car can stay on the lane

##### Load images:
+ I load the image path into X_train array, the center images takes the current streering angle, left and right images has an adjusted_angle variable to make up for the center (line 25-34)
+ In order to gauge how well the model was working, I shuffle images position then split my image and steering angle data into a training and validation set, test_size = 20%

##### PreProcess images:
+ Crop the top 60px and bottom 20px of all images
+ Resize image to **(64,64,3)** ... The reason is: when I keep the images at their original size, it takes 30 mins to train each epoch. Reducing the image size reduces paramaters which leads to faster training time
+ Radomly apply Flip & Brightness adjustment

##### Training:
+ Use a neural network model similar to the famous nVidia model. I thought this model might be appropriate because it's been widely used with great result
+ To combat the overfitting, I modified the model so that each layer has L2 regularizer and each FC layer has a dropout rate of 0.5
+ I use **generator** with batch_size = 32 to combat memory and limitation also ensure every single training step is different. This allows better generalization for the model

##### Run the simulator: several attemps to tweak 
+ The hyper-parameters (batch_size, learning-rate, epochs number ...)
+ Training dataset (2 normal laps, 2 reveser laps, recovery)
+ Image preprocess (Grayscale, Image resize, Crop different position, Random fip, Add flipped image to double the data point)

Finally the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 

![alt text][image13]

#### 3. Creation of the Training Set & Training Process
To capture good driving behavior
- I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

| Left        | Center           | Right  |
| ------------- |:-------------:| -----:|
| ![alt text][image1]      | ![alt text][image2] | ![alt text][image3] |

- Recorded another lap on track one using center lane driving but in **Reverse** order, this means the samples of turning left & right are equal. Here is an example image of Reverse center lane driving:

| Left        | Center           | Right  |
| ------------- |:-------------:| -----:|
| ![alt text][image4]      | ![alt text][image5] | ![alt text][image6] |

- After a few training & testing I realize that there's a need for **Recovery** drivng in different tricky section of the road:

| Left        | Center           | Right  |
| ------------- |:-------------:| -----:|
| ![alt text][image7]      | ![alt text][image8] | ![alt text][image9] |

- Then more training & testing reveal that my model often fails when the car approaches the bridge and goes out of the bridge, it nearly lost control. So I record a special set for **Bridge Recovery**

| Left        | Center           | Right  |
| ------------- |:-------------:| -----:|
| ![alt text][image10]      | ![alt text][image11] | ![alt text][image12] |

- I also flipped images and angles then randomly change it brightness thinking that this would helps the model generalize better (line 43 - 54)

- The total data point is **10908**. 

- sklearn.utils.shuffle the whole dataset and put 20% of the data into a validation set. (line 41)

- Used this training data for training the model. The validation set helped determine if the model was over or under fitting

- I tried 5, 10, 20 epochs, applying from Grayscale to original image size (160, 320,3) to resize image to (64,64,3) ... changing the Steps_per_epoch: len(X_train) / 4000 / 3000

![alt text][image14] 

**The ideal outcome for my model is:**
- Epochs= 20 (training time 5 mins per epoch)
- Iamge size= 64,64,3
- Steps_per_epoch = 3000

Here is the youtube [link](https://www.youtube.com/watch?v=95Nr6fPCuNc) of my model in action 
