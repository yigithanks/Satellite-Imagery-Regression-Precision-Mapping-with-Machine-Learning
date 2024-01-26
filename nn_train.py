import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tifffile
import imageio
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import cross_val_score, KFold 
from tensorflow.keras import layers, models

# determine and calculate the splitpoint index to split the image for visual examination
split_point = 600

# the raw RGB image (int16 values) is readed here
raw_rgb_image = tifffile.imread('raw_RGB_image.tif')

# the true color RGB image (int8 values) is readed here
true_color_rgb_image = tifffile.imread('true_color_RGB_image.tif')

# both images are split into train and test sets based on the split point (this is only for obtaining scores and visual examination)
image_train_16 = raw_rgb_image[:split_point]
image_train_8 = true_color_rgb_image[:split_point]

# to check the shapes of both divided images (in case if needed)

#display(raw_rgb_image.shape)
#display(image_train_8.shape)

# the images are flattened by using reshape to work with individual pixels
X_train = image_train_16.reshape((-1, 3))
y_train = image_train_8.reshape((-1, 3))

# creating the dataframes here is for only printing neatly to observe the processed data, that is why I commented them
#X_train = pd.DataFrame(X_train, columns=['R', 'G', 'B'])
#y_train = pd.DataFrame(y_train, columns=['R', 'G', 'B'])

#display(X_train)
#display(y_train)

# normalization is implemented by scaling the input features to the range [0, 1] with MinMaxScaler 
#scaler = MinMaxScaler()
#X_train_scaled = scaler.fit_transform(X_train)


# Neural network model
model2 = models.Sequential([
    layers.Input(shape=(3,)),
    layers.Flatten(),  # Flattening the input
    layers.Dense(64, activation='relu'),  # Adding a dense layer with ReLU activation
    layers.Dense(64, activation='relu'),  # Adding another dense layer
    layers.Dense(3, activation='linear')  # Output layer with linear activation for regression
])

# here we compile the 
model2.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Mean Squared Error (MSE) for regression

# here we display the model summary
model2.summary()

# training the model
model2.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# save the model
model2.save('nn_model.keras')
