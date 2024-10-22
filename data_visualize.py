# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from train import load_data, data_reshape

#%%
def visualize(X):
    print(X.shape)
    try:
        plt.imshow(X.squeeze(0).numpy(), cmap = 'gray')
    except:
        raise Exception('Data shape mismatch!')

#%%
train_set_path = '../Datasets/MNIST/split_datasets/train.csv'
X_train, y_train = load_data(os.path.join(train_set_path))
print(X_train.shape)

X_train = X_train/255.0

import matplotlib.pyplot as plt
plt.imshow(X_train[0].reshape(28,28))




'''
root_data_path = '../Datasets/MNIST/split_datasets'

train = pd.read_csv(os.path.join(root_data_path, 'train.csv'))

print(train.head())


#%%
y = train['label']
X = train.drop(['label'], axis =1)


#%% Convert to numpy
X = np.array(X)
y = np.array(y)

X = X/255

X= np.where(X <0.5, 0, 1)


#%%

X = np.reshape(X, (-1,28,28))

#%%

import matplotlib.pyplot as plt
for i in range(1,100,10):
    plt.imshow(X[i], cmap='gray')
    plt.show()

#%% Resize X
import cv2
# Assuming 'images' is your NumPy array of shape (10, 28, 28)
resized_images = []

for img in X.astype(np.uint8):
    # Make sure each image is treated as a 2D grayscale image
    if len(img.shape) == 2:
        # Resize each 28x28 image to 10x10
        resized_img = cv2.resize(img, (17, 17), interpolation=cv2.INTER_AREA)
        resized_images.append(resized_img)
    else:
        print("Unexpected image shape:", img.shape)

# Convert the list back to a NumPy array with shape (10, 10, 10)
resized_images = np.array(resized_images)

print(resized_images.shape)  # Output should be (10, 10, 10)





#%%
import matplotlib.pyplot as plt
for i in range(10):
    plt.imshow(resized_images[i], cmap='gray')
    plt.show()

'''

