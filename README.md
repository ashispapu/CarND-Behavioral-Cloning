# Behavioral Cloning - Project 3


## Overview

In this project, our goal is to  train a  car  in a simulated environment by cloning the behavior as seen during training mode. Here we're leveraging TensorFlow and Keras, a deep learning network predicts the proper steering angle given training samples.


## Files
* `model.py` - The script used to create and train the model.
* `drive.py` - The script to drive the car.
* `model.json` - The model architecture.
* `model.h5` - The model weights.

## Network

### Implementation approach : 

Here our goal is to predict the steering angle from the image captured by the cameras of the car . We're trying to map the image  pixel data from the camera to the steering angle  . It's a regression problem as we're predicting a continuous value .

I started with a simple multi layer neural network to with two convolution layers and two fully connected layers . However it couldn't generalise the prediction.It performed badly on both the tracks .

Then I thought of starting with transfer learning with vgg-16 architecture . In the mean time from the forum I came across [NVIDIA Paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and comma.ai architectures . So I thought of giving a try to already proven networks . And Its started ...

I took a look at the solution documented in the [NVIDIA Paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), in which raw pixels are mapped steering commands. Because of the similarity of the use case  I decided it would be a good starting point.  The Nvidia architecture is small compared to the previously considered architectures with only 9 layers.  After experimenting with a rough replication of the network, I found that I could train relatively fast, and because of this, I decided that did not need transfer learning to complete this project, opting to stick with the simpler Nvidia network.

After getting the initial network running, I experimented with different dropout layers and activation functions.  

For activations, I read a [Paper on ELU Activations](https://arxiv.org/pdf/1511.07289v1.pdf), which led me to experiment, comparing the training time and loss for RELU vs ELU activations.  After several trials I concluded that ELUs did indeed give marginally faster performance and lower loss.  ELU activations offer the same protection against vanishing gradiant as RELU, and in addition, ELUs have negative values, which allows them to push the mean activations closer to zero, improving the efficiency of gradient descent.

For dropout, I ran trials with values between 0.2 and 0.5 for fraction of inputs to drop, as well as which layers to include a dropout operation.  I found that my model performed poorly in autonomous mode when including dropout layers in the final fully connected layers.  My intuition here is that dropout may not be appropriate for every layer in regression problems.  In classification problems we are only concerned softmax probabilities relative to another class, so even if dropout effects the final value, it should not matter because we only care about the value relative to other classes.  With regression, we care about the final value, so dropout might have negative effects.  To avoid this dilemma, I chose l2 regularization in the fully connected layers.  Initially, this prevented the model from producing sharp turns, but was fixed after reducing the weight penalty.

### Architecture

My architecture is modeled after the network depicted in [NVIDIA Paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).  The architecture is a traditional feed-foward layered architecture in which the output of one layer is fed to the layer above.  At a high level the network consists of preprocessing layers, 5 convolutional layers, followed by 3 fully connected layers, and a final output layer.  Since we are working with a regression problem, the output layer is a single continuous value, as apposed to the softmax probabilities used for classification tasks such as traffic sign identification.

Before the first convolutional layer, a small amount of preprocessing takes place within the pipeline.  This includes pooling  and batch normalization.

Each convolitional has a 1x1 stride, and uses a 2x2 max pooling operation to reduce spatial resolution. The first three convolutional layers use a 5x5 filter, while the final two use a 3x3 filter as the input dimensionality is reduced.

For regularization, a spatial dropout operation is added after each convolutional layer.  Spatial dropout layers drop entire 2D features maps instead of individual features.

For non-linearity, ELU activationd are used for each convolutional, as well as each fully connected layer.

The output from the forth convolutional layer is flattened and fed into a regressor composed of four fully connected layers.  The fully connected layers each reduce the number of features with the final layer outputting a single continuous value.  As noted above, l2 regularization is leveraged in the fully connected layers.



| Layer (type)                                |  Output Shape           |    Param #        |  Connected to                     
|---------------------------------            |-------------------      |--------------     |------------------------- 
| maxpooling2d_1 (MaxPooling2D)               | (None, 40, 160, 3)      |  0                | maxpooling2d_input_1[0][0]             
| batchnormalization_1 (BatchNormalisation)   | (None, 40, 160, 3)      |  160              | maxpooling2d_1[0][0]                  
| convolution2d_1 (Convolution2D)             | (None, 40, 160, 24)     |  1824             | batchnormalization_1[0][0]           
| maxpooling2d_2 (MaxPooling2D)               | (None, 20, 80, 24)      |  0                | convolution2d_1[0][0]               
| spatialdropout2d_1 (SpatialDropout)         | (None, 20, 80, 24)      |  0                | maxpooling2d_2[0][0]             
| convolution2d_2 (Convolution2D)             | (None, 20, 80, 36)      |  21636            | spatialdropout2d_1[0][0]                      
| maxpooling2d_3 (MaxPooling2D)               | (None, 10, 40, 36)      |  0                | convolution2d_2[0][0]           
| spatialdropout2d_2 (SpatialDropout)         | (None, 10, 40, 36)      |  0                | maxpooling2d_3[0][0]                
| convolution2d_3 (Convolution2D)             | (None, 10, 40, 48)      |  43248            | spatialdropout2d_2[0][0]            
| maxpooling2d_4 (MaxPooling2D)               | (None, 5, 20, 48)       |  0                | convolution2d_3[0][0]               
|spatialdropout2d_3 (SpatialDropout)          | (None, 5, 20, 48)       |  0                | maxpooling2d_4[0][0]                 
|convolution2d_4 (Convolution2D)              | (None, 5, 20, 64)       |  27712            | spatialdropout2d_3[0][0]                    
| maxpooling2d_5 (MaxPooling2D)               | (None, 3, 10, 64)       |  0                | convolution2d_4[0][0]                     
| spatialdropout2d_4 (SpatialDropout)         | (None, 3, 10, 64)       |  0                | maxpooling2d_5[0][0]                
| convolution2d_5 (Convolution2D)             | (None, 3, 10, 64)       |  36928            | spatialdropout2d_4[0][0]                  
|maxpooling2d_6 (MaxPooling2D)                | (None, 2, 5, 64)        |  0                | convolution2d_5[0][0]  
| batchnormalization_2 (BatchNormalisation)   | (None, 2, 5, 64)        |  8                | maxpooling2d_6[0][0]
| spatialdropout2d_5 (SpatialDropout)         | (None, 2, 5, 64)        |  0                | batchnormalization_2[0][0]
| flatten_1 (Flatten)                         | (None, 640)             |  0                | spatialdropout2d_5[0][0] 
| dense_1 (Dense)                             | (None, 100)             |  64100            | flatten_1[0][0]
| dense_2 (Dense)                             | (None, 50)              |  5050             | dense_1[0][0] 
| dense_3 (Dense)                             | (None, 10)              |  510              | dense_2[0][0]
| dense_4 (Dense)                             | (None, 1)               |  11               | dense_3[0][0]    
||||
| Total params: 201,187
| Trainable params: 201,103
| Non-trainable params: 84



## Data Collection

Here , I've trained the model using the data provided by the udacity  with data augmentation techniques .

## Training dataset collection and generation 


The data collected via keyboard input was very disturbed as a joystick was not available .

It did not correctly record the correct maneuvers the model should make to recover from the edges of the road. 

To overcome this , I started data augmenting  on the existing data provide by Udacity using several techniques, as shown below. Original Sample:

![ScreenShot](images/sample_feature.jpg)

+ Shearing
![ScreenShot](images/random_shear.jpg)

+ Cropping
![ScreenShot](images/random_crop.jpg)

+ Flipping
![ScreenShot](images/random_flip.jpg)

+ Adjusting Brightness 
![ScreenShot](images/random_brightness.jpg)                 

### Image Generator

The entire set of images used for training would consume a large amount of memory.  A python generator is leveraged so that only a single batch is contained in memory at a time.

### Image Preprocessing


First the image is cropped above the horizon to reduce the amount of information the network is required to learn.  Next the image is resized to further reduce required processing.  Finally normalization is applied to each mini-batch.  This helps keep weight values small, improving numerical stability. In addition since our mean is relatively close to zero, the gradient descent optimization will have less searching to do when minimizing loss.

### Network Output

Once the network is trained, the model definition as well as the trained weights are saved so that the autonomous driving server can reconstruct the network and make predictions given the live camera stream.

Now we can run the simulator in autonomous mode and start the driver server.

```
python drive.py model.json
```

The autonomous driving server sends predicted steering angles to the car using the trained network.  Here we can test how well the model performs.  If the car makes mistakes, we return to training mode to collect more training data.

