import string
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

DIM = 28
EPOCHS = 10
LR = 1e-3

# Read data
data = pd.read_csv(r"data.csv").astype('float32')
X = data.drop('0',axis = 1)
y = data['0']

# Generate train and test sets
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2)
train_x = np.reshape(train_x.values, (train_x.shape[0], DIM, DIM))
test_x = np.reshape(test_x.values, (test_x.shape[0], DIM, DIM))
print("Train data shape: ", train_x.shape)
print("Test data shape: ", test_x.shape)

# Simple dictionary of all letters
word_dict = {i: c for i, c in enumerate(string.ascii_uppercase)}
print(word_dict)

# Reshape 
train_X = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
print("New shape of train data: ", train_X.shape)
test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
print("New shape of train data: ", test_X.shape)
train_y = to_categorical(train_y, num_classes = len(word_dict), dtype='int')
print("New shape of train labels: ", train_y.shape)
test_y = to_categorical(test_y, num_classes = len(word_dict), dtype='int')
print("New shape of test labels: ", test_y.shape)

# Create the simple convolutional neural network
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(DIM,DIM,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(len(word_dict), activation="softmax"))

# Small summary of the model architecture
model.summary()

# Train
model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_X, train_y, epochs=EPOCHS, validation_data=(test_X,test_y))

model.save(r'model.h5')

# Some metrics
print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])