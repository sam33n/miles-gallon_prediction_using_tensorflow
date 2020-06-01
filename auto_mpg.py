import tensorflow as tf
from tensorflow import keras , layers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np


data_file = keras.utils.get_file('auto-mpg.data',
                                 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')

columns_name = ['Mpg' , 'Cylinders', 'Displacement', 'Weight' , 'Acceleration', 'Model_year', 'Origin']

dataset = pd.read_csv(data_file, names=columns_name, skipinitialspace=True, sep=' ', comment='\t', na_values='?')

origin = dataset.pop('Origin')

dataset['USA'] = (origin==1) * 1.0
dataset['China'] = (origin==2) * 1.0
dataset['Europe'] = (origin==3) * 1.0

dataset.dropna(inplace=True)


#graphical representation
#sns.heatmap(dataset[['Mpg' , 'Cylinders', 'Displacement', 'Horsepower', 'Weight' , 'Acceleration', 'USA' , 'China', 'Europe']].corr(), square=True)
#sns.regplot(x=dataset['Mpg'], y=dataset['Cylinders'], data=dataset)
#plt.show()

X = dataset.iloc[:, 1:10]
y = dataset.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#data scaling and normalization
X_train= preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

#training the model using tensorflow keras API
model = keras.Sequential([
                         layers.Dense(32, activation=tf.nn.relu, input_shape=[len(X.keys())]),
                         layers.Dense(32, activation=tf.nn.relu),
                         layers.Dense(1)
                         ])


model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['mae', 'accuracy']
              )


X_train, y_train = np.array(X_train) , np.array(y_train)
print(X_train)
#history = model.fit(X_train, y_train, epochs=500, verbose=2)

#mae, loss, score = model.evaluate(X_test, y_test)
#print(score * 100)
#print(loss)

#for prediction
#predictor = model.predict(X_test)
#print(predictor)






