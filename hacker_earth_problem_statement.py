import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras, layers
from sklearn.preprocessing import OneHotEncoder
import os


df = pd.read_csv('Downloads/dataset/train.csv')
df2 = pd.read_csv('Downloads/dataset/test.csv')

PATH = os.getcwd()
IMG_SIZE = 150

X_train = []

def training_dataset():
    for item in range(len(df)):
        image_path = PATH + '/Downloads/dataset/train/'
        image_array = cv2.imread(os.path.join(image_path, df.Image[item]), cv2.IMREAD_GRAYSCALE)
        new_array  = cv2.resize(image_array, (IMG_SIZE,IMG_SIZE))
        X_train.append([new_array, df.target[item]])
    return X_train


X_test = []

def testing_dataset():
    for item in range(len(df2)):
        image_path = PATH + '/Downloads/dataset/test/'
        image_array = cv2.imread(os.path.join(image_path, df2.Image[item]), cv2.IMREAD_GRAYSCALE)
        new_array  = cv2.resize(image_array, (IMG_SIZE,IMG_SIZE))
        X_test.append(new_array)
    return X_test

x_test = testing_dataset()

X = []
y = []
for feature, label in training_dataset():
    X.append(feature)
    y.append(label)


X_train = np.array(X).reshape(-1,150,150,1)
train_label = np.array(y).reshape(-1,1)
label_encoded = OneHotEncoder().fit_transform(train_label)
y_train = label_encoded.A
X_train = X_train/255.0


model = keras.Sequential([
                        layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=X_train.shape[1:]),
                        layers.MaxPooling2D(pool_size=(2,2), strides=(1,1)),

                        layers.Conv2D(64,(3,3), activation=tf.nn.relu),
                        layers.MaxPooling2D(pool_size=(2,2), strides=(1,1)),

                        layers.Flatten(),
                        layers.Dense(64, activation=tf.nn.relu),
                        layers.Dense(8, activation=tf.nn.softmax)
                      ])


model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size=4, validation_split=0.1, epoch=5)













