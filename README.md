# Satellite-Imagery-Regression-Precision-Mapping-with-Machine-Learning
Welcome to the Satellite Image Color Mapping repository! This project aims to precisely map higher precision int16 values from raw RGB satellite images to lower precision int8 values in true color RGB images using two regression models: linear regression and neural network.

In this study, two regression models have been employed to map a 16-bit RGB image
to a 8-bit RGB image. What is expected from the models, namely a linear regression
model and a neural network model, is to be capable of mapping the higher precision int16
values from the raw RGB image to the lower precision int8 values in the true color RGB
image. Both models take 3 int16 values as their input and should return 3 int8 values as
output. In the report, the process of constructing the models is explained along with
reasonings, and some questions are answered.

## Machine Learning Design Methodology

The design methodology of the ML models consists of several steps. Since the models that
we are supposed to work with are provided, preprocessing of the data is maintained in a
compatible way for both Linear Regression and Neural Network models.
The preprocessing stage starts with reading the 8 and 16-bit RGB images provided as
train data. To obtain the numeric values in for each pixel and channels in the images,
‘imread’ function of ‘tifffile’ library is used. After importing, the shapes of the image
variables are flattened to a compatible format for models by using ‘reshape’ function.
After reshaping, we have 3 values for each RGB channels for each pixel in the image
variables. The reshaped 16-bit image constructs the features of our data, while the
reshaped 8-bit image forms the target for our model.

Before fitting and predicting by using models, one more step of normalization is performed.
Normally, normalization is a highly used scaling method since it scales all the features
into the same range and thus prevent the possibility of some large-scale features having
more impact compared to the lower-scale ones. However, in this project, all features that
we described are in the same range (0-255 for each of the R, G, and B channels) and thus,
they could have similar weights on the target variable. Yet, I wanted to add and try
normalization thinking that the satellite images are generally consist of green and brown
blocks, which makes some pixels more present in the dataset compared to the others. To
examine if such a cause would affect the impact of features on the target variable, I
obtained scores both with and without normalization, and they cases did not differ more
than 0.01 in the R2 and SSIM scores.

For resampling, I tried 2 of 3 commonly used resampling methods: train-test split and
Cross-Validation. For train-test split process, I tried two alternative methods, both of
which did not affected the performance of the model. Firstly, I used ‘train_test_split’
from ‘sklearn.model_selection’, which is the most commonly used train test split method
for ML modeling. However, it is noticeable that train-test-split splits the dataset randomly 
and for images, this means that disrupts the order of the pixels in the image for both train
and test sets. Later, I thought that order of the pixels might not have a considerable effect
on the model performance since the model evaluates the values pixel by pixel, I still
wanted to use a different train-test splitting method since in the actual model, we are
training the model with an image that is not shuffled. With that purpose, I simply divided
the 8 and 16-bit images vertically from the exact same pixels and kept the larger ones as
the training set while keeping the smaller ones as the test set. In that way, the structure
of the images is preserved, and the train and test sets are more close to the real cases that
could be studied. Moreover, I wanted to use this splitting to be able to visually examine
my models and be able to see the differences between y_pred and y_test after running
the models. The train and test splits for this method can be seen below.

![train](https://github.com/yigithanks/Satellite-Imagery-Regression-Precision-Mapping-with-Machine-Learning/assets/125910884/6c94afa9-0a04-4da9-a74b-a96b0c100b8a)

![Train image after splitting](image_url)
![Test image after splitting](image_url)

To ensure the reliability of the model, another resampling method of Cross-Validation is
used as well. Unlike the train-test split method explained above, for the Cross-Validation,
the default K-Fold splitting is used, meaning that I did not try to preserve the pixel orders,
thus the image integrity in this resampling. As mentioned above, I did not encounter any
significant difference between the scores obtained by two methods. For Cross-Validation,
a mean R2 score of 0.9987 is obtained with a standard deviation of 0.0018 for K=100
folds. This demonstrated that the outcomes are stable enough and does not vary from
sample to sample in many observations.

For model performance metrics, I worked with MSE, MAE, R2 and SSIM scores. I wanted
to add SSIM (Structural Similarity Index Measure) since it is a commonly used metrics 
for comparing two images of the same size and shape. It evaluates the predicted and the
true images based on the luminance, contrast, and structure of them and calculate a
similarity score. Just as in R2, an SSIM score of 1 refers to a perfect fit. To have the
performance metrics in the same range and since Cross-Validation by default uses R2
score, I decided to continue with both R2 and SSIM scores as the model performance
metrics. For SSIM scores, Linear Regression gives 0.998 while Neural Network model
performed between 0.997-0.999.

In the neural network model, the input layer has three neurons, followed by a flattening
layer to convert the 2D input to 1D. Two hidden layers with 64 neurons each and ReLU
activation is used for non-linearity, and the output layer with linear activation produces
three continuous values corresponding to RGB channels. The model is compiled using the
Adam optimizer and mean squared error (MSE) as the loss function. During training, the
model is trained for 10 epochs with a batch size of 32, and 20% of the training data is
used for validation (I tried different numbers of epoch but as the number goes up, time
required to run the model increased a lot as well). After predictions, a short postprocessing is performed by clipping values to [0, 255], rounding to integers, and then SSIM
score calculation is held on image data.

I found the Linear Regression model results satisfying after examining the model by both
visually and by two performance metrics. Even though there are some pixels that the
model was not able to predict correctly (it is noticeable that almost all of them appeared
as blue pixels after incorrect predictions), overall the input image is obtained with a very
high similarity and the colors and patterns of the image is highly presented in the predicted
output.

![y_test for Visual Examination](image_url)
![y_pred Visualized (Linear Regression)](image_url)

I tried both Ridge and Lasso regularization to see if it will improve the model performance.
Both Ridge and Lasso regularizations add penalty terms to the coefficients and helps to 
calibrate ML models to minimize the adjusted loss function, and therefore prevents
overfitting or underfitting. By changing the values of the penalty function, we are
controlling the penalty term. The higher the penalty is, which reduces the magnitude of
coefficients, and eventually shrinks the parameters. Therefore, it is used to prevent
multicollinearity, and it reduces the model complexity by coefficient shrinkage.
However, as mentioned above, we do not have many features that have different scales,
and this cause all three channels to have similar effects on the target variable. That is
why, I believe, using regularization will not cause a significant increase in the model
performance. Yet, I wanted to try and see the effect of both regularization methods. Both
models performed best for an alpha value of 0.01, and the maximum score obtained by
using regularization did not go over 0.9983.

The most significant difference between the Linear Regression model and the Neural
Network model was the runtime. For the data size of our case, Linear Regression takes
less then 5 seconds to run and bring the outputs while Neural Network model takes 90-
100 seconds to finish running on average. This could affect one’s preference among both
methods while working with larger datasets.

Even though both models performed well and obtained scores around 0.99, I can say that
neural network models are able to minimize the loss function and during some runs, it
was able to obtain scores of 0.999. I think using Neural network modeling with enough
number of runs could lead a project to bring better outputs if the trade-off between time
spent and score seems reasonable. Additionally, Linear regression is a simpler model to
perform with minimum numbers of parameters, which makes it more practical to
implement. On the other hand, Neural Networks have more hyperparameters and thus it
is more flexible compared to the Linear Regression. For more complex cases, I believe it
would be able to learn complex patterns better compared to the Linear Regression
modeling.

The docker image is created and pushed to the Docker Hub. It can be run by the username
and image name in the following line:
docker run -p 4000:80 yigithanks/doktar_case:latest
