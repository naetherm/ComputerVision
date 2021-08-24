import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BATCH_SIZE = 128
EPOCHS = 1

train = pd.read_csv('./sign_mnist_train.csv') 
test = pd.read_csv('./sign_mnist_test.csv')

labels = train['label'].values
images = train.drop('label', axis=1)
images = images.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
# convert 28x28 array into a 1d array of size 784
images = np.array([i.flatten() for i in images])
labels.shape

y_test = test['label'].values
test = test.drop('label', axis=1)
test = test.values
test = np.array([np.reshape(i, (28, 28)) for i in test])
x_test = np.array([i.flatten() for i in test])


x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.3)

# # add bias
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

# Normalize the data
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

n_classes = 26

y_train = utils.to_categorical(y_train, n_classes)
y_val = utils.to_categorical(y_val, n_classes)
y_test = utils.to_categorical(y_test, n_classes)


model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=n_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history1 = model.fit(
  x_train, 
  y_train, 
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_data=(x_val,y_val))

print("Test Accuracy: " , model.evaluate(x_test, y_test)[1]*100 , "%")
y_pred = model.predict(x_test)
y_preds = []
for i in range(len(y_pred)):
  y_preds.append(np.argmax(y_pred[i]))

accuracy_score(y_test, y_preds)