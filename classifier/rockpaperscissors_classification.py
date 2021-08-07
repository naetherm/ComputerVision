import os
import pathlib

import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.losses import CategoricalCrossentropy

CLASSES = ['rock', 'paper', 'scissors']
AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 1024
BUFFER_SIZE = 1024
EPOCHS = 200
FILE_PATTERN = './rps/*/*.png'


def load_data(image_path, target_size=(32, 32)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, np.float32)
    image = tf.image.resize(image, target_size)

    label = tf.strings.split(image_path, os.path.sep)[-2]
    label = (label == CLASSES)  # One-hot encode.
    label = tf.dtypes.cast(label, tf.float32)

    return image, label


input_layer = Input(shape=(32, 32, 1))
x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(input_layer)
x = ReLU()(x)
x = Dropout(rate=0.5)(x)

x = Flatten()(x)
x = Dense(units=3)(x)
output = Softmax()(x)

model = Model(inputs=input_layer, outputs=output)


def prepare_data(
    dataset_path,
    buffer_size,
    batch_size,
    shuffle=True):
    dataset = (
        tf.data.Dataset
           .from_tensor_slices(dataset_path)
           .map(load_data,
                num_parallel_calls=AUTOTUNE))

    if shuffle:
        dataset.shuffle(buffer_size=buffer_size)

    dataset = (
        dataset
            .batch(batch_size=batch_size)
            .prefetch(buffer_size=buffer_size))

    return dataset


file_pattern = str(FILE_PATTERN)
dataset_paths = [*glob.glob(file_pattern)]

train_paths, test_paths = train_test_split(dataset_paths, test_size=0.2, random_state=999)
train_paths, val_paths = train_test_split(train_paths, test_size=0.2, random_state=999)

train_dataset = prepare_data(train_paths, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)
validation_dataset = prepare_data(val_paths, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, shuffle=False)
test_dataset = prepare_data(test_paths, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, shuffle=False)

model.compile(
    loss=CategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS)

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Loss: {test_loss}, accuracy: {test_accuracy}')
