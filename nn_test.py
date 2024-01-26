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
import tensorflow as tf
from tensorflow.keras import layers, models

# determine and calculate the splitpoint index to split the image for visual examination
split_point = 600

# the raw RGB image (int16 values) is readed here
raw_rgb_image = tifffile.imread('raw_RGB_image.tif')
true_color_rgb_image = tifffile.imread('true_color_RGB_image.tif')

# both images are split into train and test sets based on the split point (this is only for obtaining scores and visual examination)
image_test_8 = true_color_rgb_image[split_point:]
image_test_16 = raw_rgb_image[split_point:]

# to check the shapes of both divided images (in case if needed)

#display(raw_rgb_image.shape)
#display(image_train_8.shape)
#display(image_test_8.shape)

# the images are flattened by using reshape to work with individual pixels
X_test = image_test_16.reshape((-1, 3))

# load the model
model2 = tf.keras.models.load_model('nn_model.keras')

y_pred2 = model2.predict(X_test).clip(0,255)
y_pred_rounded2 = np.round(y_pred2).astype(np.uint8)
predicted_rgb_image2 = y_pred_rounded2.reshape(image_test_8.shape)

# here we calculate the SSIM score for our predicted RGB image (Neural Networks)
ssim_value2, _ = ssim(predicted_rgb_image2, image_test_8, full=True, win_size = 3)

print(f'SSIM score for Neural Networks : {ssim_value2}')