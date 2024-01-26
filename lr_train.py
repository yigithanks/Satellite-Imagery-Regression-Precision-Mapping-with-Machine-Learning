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


# here Cross-validation is used with n_splits = 100. Other numbers of n_splits are also checkhed as well and they all perform similar scores.
kf = KFold(n_splits=100, shuffle=True, random_state=42) 
reg = LinearRegression() 

X = raw_rgb_image.reshape((-1, 3))
y = true_color_rgb_image.reshape((-1, 3))

cv_results = cross_val_score(reg, X, y, cv=kf, scoring='r2')

print(f'CV mean: {np.mean(cv_results)}, CV standard deviation: {np.std(cv_results)}')


# linear regression model is assigned to the model variable
model = LinearRegression()

# training the model
model.fit(X_train, y_train)

#save model
joblib.dump(model, "lr_model.sav")


