from classification_models.keras import Classifiers
import glob
import math

import numpy as np
import pandas as pd
import os
import shutil
#import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator, array_to_img
from sklearn.preprocessing import LabelEncoder

from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
import keras

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

#matplotlib inline
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.training.input import batch




IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

train_path = "/home/jscesar/datasets/muct_train"
validation_path="/home/jscesar/datasets/muct_validation"

train_datagen = ImageDataGenerator().flow_from_directory(
    directory=train_path,
    target_size=IMG_DIM,
    batch_size=32,
    class_mode="categorical",
    interpolation="nearest",

)
val_datagen = ImageDataGenerator().flow_from_directory(
    directory=validation_path,
    target_size=IMG_DIM,
    batch_size=32,
    class_mode="categorical",
    interpolation="nearest"
)

ResNeXt50, preprocess_input = Classifiers.get('resnext50')
model = ResNeXt50(
            include_top = False,
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
            weights='imagenet')

output = model.layers[-1].output
output = Flatten()(output)

model = Model(model.input, outputs=output)
model.trainable = True

for layer in model.layers:
    layer.trainable = False

new_model = Sequential()
new_model.add(model)
new_model.add(Dense(512, activation='relu', input_dim=IMG_DIM))
new_model.add(Dropout(0.3))
new_model.add(Dense(512, activation='relu'))
new_model.add(Dropout(0.3))
new_model.add(Dense(278, activation='sigmoid'))

new_model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy'])

callback = EarlyStopping(monitor='loss', patience=3)

history_1 = new_model.fit_generator(train_datagen,
                                  steps_per_epoch=3336/32,
                                  epochs=30,
                                  validation_data=val_datagen,
                                  validation_steps=834/32,
                                  verbose=1,
                                  callbacks=[callback])

new_model.save('modeloResnextMuct.h5')

