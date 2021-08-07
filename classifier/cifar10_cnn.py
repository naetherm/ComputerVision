import numpy as np
import tensorflow as tf
from tensorflow import keras

# Parameters
BATCH_SIZE = 128
BUFFER_SIZE = 128
EPOCHS = 10
HEIGHT = 24
WIDTH = 24

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

def process_files(image, label):
    img = tf.image.resize_with_crop_or_pad(image, WIDTH, HEIGHT)
    img = image / 255
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img,max_delta=0.1)
    img = tf.image.random_contrast(img,lower=0.5, upper=0.8)
    return img, label
    

train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_P = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).map(process_files)
test_P = test.batch(BATCH_SIZE).map(process_files)

model = keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=[32,32,3], filters=32, kernel_size=3, padding='SAME', activation="relu", kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME',activation="relu", kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation="relu", kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation="relu", kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation="relu", kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation="relu", kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(name="FLATTEN"),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=10, activation="softmax")
])
    
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(train_P, validation_data=test_P, epochs=EPOCHS)
