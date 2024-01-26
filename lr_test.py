import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tifffile
import imageio
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import cross_val_score, KFold 
import joblib

# loading the model
model = joblib.load("lr_model.sav")

# determine and calculate the splitpoint index to split the image for visual examination
split_point = 600

# the raw RGB image (int16 values) is readed here
raw_rgb_image = tifffile.imread('raw_RGB_image.tif')

# the true color RGB image (int8 values) is readed here
true_color_rgb_image = tifffile.imread('true_color_RGB_image.tif')

# both images are split into train and test sets based on the split point (this is only for obtaining scores and visual examination)
image_test_16 = raw_rgb_image[split_point:]
image_train_16 = raw_rgb_image[:split_point]
image_test_8 = true_color_rgb_image[split_point:]

# to check the shapes of both divided images (in case if needed)

print(raw_rgb_image.shape)
print(image_test_8.shape)

# the images are flattened by using reshape to work with individual pixels
X_test = image_test_16.reshape((-1, 3))
X_train = image_train_16.reshape((-1, 3))

# creating the dataframes here is for only printing neatly to observe the processed data, that is why I commented them
#X_test = pd.DataFrame(X_test, columns=['R', 'G', 'B'])

#display(X_test)


# making predictions on the test set
y_pred = model.predict(X_test).astype(np.uint8)

# rounding the predicted float values to the closest integers as int8 values
y_pred_rounded = np.round(y_pred).astype(np.uint8)

#display(y_pred_rounded)

predicted_rgb_image = y_pred_rounded.reshape(image_test_8.shape)

#display(image_test_8.shape)
#display(predicted_rgb_image.shape)

# here we calculate the SSIM score for our predicted RGB image (Linear Regression)
ssim_value1, _ = ssim(predicted_rgb_image, image_test_8, full=True, win_size = 3)

print(f'SSIM score for Linear Regression: {ssim_value1}')

tifffile.imwrite('y_pred1.tif', predicted_rgb_image)
tifffile.imwrite('y_test1.tif', image_test_8)
