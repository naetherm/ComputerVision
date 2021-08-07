import os
import pathlib

import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import *

BATCH_SIZE = 32
EPOCHS = 20
FILE_PATTERN = './SMILEsmileD-master/SMILEs/*/*/*.jpg'

def load_data(image_paths):
    images = []
    labels = []

    for image_path in image_paths:
        image = load_img(image_path, target_size=(32, 32),
                         color_mode='grayscale')
        image = img_to_array(image)

        label = image_path.split(os.path.sep)[-2]
        label = 'positive' in label
        label = float(label)

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


input_layer = Input(shape=(32, 32, 1))
x = Conv2D(filters=20,kernel_size=(5, 5), padding='same', strides=(1, 1))(input_layer)
x = ELU()(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.4)(x)

x = Conv2D(filters=50, kernel_size=(5, 5), padding='same', strides=(1, 1))(x)
x = ELU()(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Dropout(0.4)(x)

x = Flatten()(x)
x = Dense(units=500)(x)
x = ELU()(x)
x = Dropout(0.4)(x)

output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)

FILE_PATTERN = str(FILE_PATTERN)
dataset_paths = [*glob.glob(FILE_PATTERN)]

X, y = load_data(dataset_paths)

X /= 255.0 # Normalize
total = len(y)
total_positive = np.sum(y)
total_negative = total - total_positive

(X_train, X_test,
 y_train, y_test) = train_test_split(
    X, 
    y,
    test_size=0.2,
    stratify=y,
    random_state=999)
(X_train, X_val,
 y_train, y_val) = train_test_split(
    X_train, 
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=999)

model.compile(
    loss='binary_crossentropy', 
    optimizer='rmsprop', 
    metrics=['accuracy'])

model.fit(
    X_train, 
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight={
          1.0: total / total_positive,
          0.0: total / total_negative
    })
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {test_loss}, accuracy: {test_accuracy}')
