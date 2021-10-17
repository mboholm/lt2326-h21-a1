# Summary
**Name: Max Boholm** (gusbohom)
This repository contains my code for Assignment 1, (course) LT2326, autumn 2021. The repository contains the following files:

*    `lt2326-h21-a1.ipynb` this is the file which implements the solution for Assignment 1
*    `obsolete.ipynb` obsolete code, *please ignore*
*    `utils_DimensionTransformationsCNNs.ipynb` contains code for finding workable transformations of dimensions in CNNs; *can be ignored*
*    `xxx.pt` the best model, **trained for images reduced to 512 x 512 and for a target size of 512 x 512**

## A note on performance
The trained models (including the best one) does not perform well in terms of F1 (recall is very low). In terms of accuracy and Mean Square Error, both models are at first glance impressive, but this must be interpreted in relation to the task at hand. For this task, a baseline model of "always-predict-zero-probability-for-a-pixel" will in general get high scores of accuracy and low for MSE. When trained, both models seem to converge to the "always-predict-zero-probability-for-a-pixel" strategy, especially with large minibatches, e.g. of size 128. Possibly, there is a local minimum of the loss function associated with "always-predict-zero-probability-for-a-pixel" weigths. The relevant question to raise is then how to create a model that does not get stuck at that minimum... a question for which I have no answer (more data is proably part of the answer). 

## On changing parameters
The code has been written with flexibility in mind. That is, rather than "hard-coded" parameters, there is much freedom in setting parameters of the pipline by changing parameters of the functions. Mostly this is done by defining variables initially and then use that variable downstream. In the notebook there is a cell for the main *meta variables* of the pipline: directories of images and utility files, downsizing of images and dataset, and which cuda device (or other) to use. **It is highly recommended to change these variables here and not elsewhere in as it might mess up the steps of the pipline**.

Changing size of input and output **MUST** follow division by 2^*n*: 2048 / 2 = 1024, 2048 / 2² = 512, 2048 / 2³ = 256. The pipline, icluding model layers and parameters, has been built such that if images sizes is selected from this set of 2048 / 2^*n* and defined under *Meta variables* the code should execute without errors lower down the pipline (i.e., data preparation, model definitions, training and evaluation).

Hyperparameters for every model, e.g. number of epochs, size of batches, learning rate, can be set and modified after each model definition. 

For evaluation, there is one key parameter to keep in mind: the consideration of threshold metrices (e.g. accuracy), or not. Considering these metrices takes a lot of time and is therefore optional. For outputs of 512 x 512, or less, the time for computing these metrices are acceptable (about 20 miutes), while for larger outputs, these calculations takes a lot of time. 

Also detachment from cuda and moving to cpu of performance calcualtions is possible. CUDA memory has been a struggle working with this assignment. 


## On input and output downscaling
**There will be memory issues with large input images** (i.e. keeping images in their orignal 2048 x 2048 sizes). Also, **large outputs cost time**, for example, creating arrays/matrices corresponding to polygon boxes and to evaluate accuracy, recall, etc.  

I have mainly worked with images **downsized to 512 x 512 and for outputs at the same size**, but it should be possible to run the code with 512 input and 1024 or 2048 output, but it will take longer time (due to building polygon-matrices representations and threshold-based evaluations, e.g. accuracy). The code will not handle images which are not downsized well (it will complain about CUDA memory and also it will have problems with standardization of data).   

## Data preparation

    1.For every file (image) in `/scratch/lt2326-h21/a1/images` on mltgpu, check if it is in the CTW training; if so, keep it (`only_train()`). If a restriction is selected for the dataset, only this subsample will be processed further (`shorty()`).
    2.For the selected files, map those files with their CTW annotations.
    3.Build the dataset, consisting of the images and their "labels". Images are downsized as defined by `rescale_input_to`. Labels are construed as matrices of 0s and-1s as defined by the CTW annotations for polygon boxes. Each label is sized *n* x *n*, where *n* = 2048 or as `rescale_output_to` (e.g. 512 x 512). 
    4.Image data is standardized.
    5.Image data and 0s-and-1s label matrices are transfered to torch tensors on the specified device. Image data is permuted (channel, heigth, width). 
    6.Data is splitted into train data and test data. 
    7.A dataloader is defined.
  

## Models
There are two models:

The first model uses three convolutional layers with randomized ReLU and maxpooling to compress data. The compressed (encoded) representation is then upsampled to output size, flattened and sent to a sigmoid. The first layer uses a `BatchNorm2d()` which in theory should counteract exploding and vanishing gradients (unclear if that is really required here, as the model is not that deep). Randomized ReLU (`RReLU()`) was used as an attempt to improve on the problem that the models seem to converge to predict "not character box" for *every* pixel (local minimum of the loss function?), but it does not seem to help.

The second model, is a convolutional autoencoder, which is trained to encode the image and then decode it  it as 0s-and-1s matrix representing the character boxes. Like the first model, the second model uses three layers of convolution and maxpooling to encode the images (enocoder). Then, with inspiration from https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac, the decoder uses five layers of transposed convolutions (`ConvTranspose2d`) with `BatchNorm2d()`. 

Part of the motivation of selecting hyperparameters for convolutional layers and upsampling has been that they should work with different input and output sizes without redefining the models. That is, images with dimension of, for example, 512 can be processed to an output dimension of 2048 or 1024 using the *same* model definition. This makes the code flexible for testing different different dimensions of input and output, for example, during development. However, from a machine learning perspective this approach is admittedly *ad hoc*. For example, there is little theoretical motivation for why we should use larger kernels, strides and padding for smaller inputs of the decoder in the second model.

With CUDAs being busy during working hours and many evenings, experimenting with variations of hyperparameters has not been somewhat struggeling. There is not always a clear rationale for the choice of parameters here. Previous examples on the web has been the main inspiration. The main ideas implemented are: larger kernels and strides of convolutions (i.e. more compression) are followed by smaller more detailed representations; ReLU for additional non-linearity; models cannot contain too many parameters (CUDA memory will run out); `BatchNorm2d` to help with vanisin gradient problems. I have tested two loss functions, i.e. `MSELoss` and `BCELoss`, but did not get any significant improvements with one or the other. 

Run with 512 x 512 input images the models does not take up too much memory. 


## Evaluation
Two basic types of evaluation metrics are considered:

1. The "continious" ("analog") metric of *mean squared error*.
2. Threshold-based ("digital", frequency-based) metrics, assuming a treshold *t* for a classfier *C*, such that for every pixel *x*, if the probaility predicted for *x* (i.e. *p(x)*) is greater than *t*, then *C(x)* = 1, if not, *C(x)* = 0. Represented by a threhold-classification, true positives (TP), false positives (FP), true negatives (TN) and false neagtives (FN) can be calculated and therfore also standard measures of *accuracy*, *recall*, *precision* and *F1*. 

Both types of metrics (analog and digital) can be measured for the model's performance on *individual* images. However, general measures of the model's performance on the *complete* test set must be considered. For this, two approaches are used:

*    A pooled approach: the evaluation metrics are calculated for the concatenation of predictions for every image of the test set in relation to the concatenation of every true label (pixel map of polygon boxes). 

`Metric([PredictionImage-1 + ... + PredictionImage-n], [TruthImage-1 + ... + TruthImage-n])` (where `+` here stands for concatenation, not addition). 
*    An averaging approach: taking the mean and standard deviation of a particular metric calculated for individual images 

`Mean([Metric(image-1), ..., Metric(image-n)])`    

## Qualitative error analysis
As noted above, the models seems to learn that in general predicting "not box" for a pixel decreases loss. As such the models in general predict 0s. As a general strategy this will be most accurate and least erroneous for images with few boxes and limited "box areas", but worse on images with many and large character boxes.
