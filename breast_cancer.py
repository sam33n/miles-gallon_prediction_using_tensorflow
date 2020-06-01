import tensorflow as tf
from tensorflow import keras, layers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix



data_file = keras.utils.get_file('breast_cancer.data',
                                 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data')


columns_name = ['Class', 'age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiate']
df = pd.read_csv(data_file, names=columns_name, na_values='?', skipinitialspace=True)

df.dropna(inplace=True)

ordinal_encode = OrdinalEncoder().fit_transform(df[['Class', 'age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiate']])

#visualize
#plt.scatter(ordinal_encode[:,7], ordinal_encode[:,0])
#plt.xlabel('breast')
#plt.ylabel('Class')
#plt.show
X = ordinal_encode[:, 1:]
y = ordinal_encode[:, 0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)
input_shape = X_train[0].shape
print(input_shape)
model = tf.keras.Sequential([
                            layers.Dense(64, activation=tf.nn.relu, input_shape=input_shape),
                            layers.Dense(64, activation=tf.nn.relu),
                            layers.Dense(1, activation=tf.nn.sigmoid)
                           ])


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=['binary_crossentropy'],
              metrics=['accuracy']
              )

#history = model.fit(X_train, y_train, epochs=100, verbose=2, validation_split=0.2)
loss, score = model.evaluate(X_test, y_test)
#print(score)
predictor = model.predict(X_test)
#print(predictor)


#performance evaluation

#logre_clf = LogisticRegression(C=1.0, random_state=0, n_jobs=-1) 78%
#dtree_clf = DecisionTreeClassifier()
#score = cross_val_score(dtree_clf, X_train, y_train, cv=5, scoring='accuracy')
