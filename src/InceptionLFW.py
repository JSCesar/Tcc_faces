from classification_models.keras import Classifiers
import glob
import math
from keras.regularizers import l2
import numpy as np
import pandas as pd
import os
import shutil
# import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator, array_to_img
from sklearn.preprocessing import LabelEncoder

from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
import keras

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, GlobalAveragePooling2D
from keras.models import Sequential
from keras import optimizers

from datetime import datetime

# matplotlib inline
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.training.input import batch

IMG_WIDTH = 300
IMG_HEIGHT = 300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
train_size = 12 * 4 * 64
validation_size = 4 * 4 * 64
batch_size = 16

date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
model_name = f"lfw_resnet_{date}"
model_path = '/home/jscesar/Tcc/logs/' + model_name

if not os.path.exists(model_path):
    os.mkdir(model_path)

train_path = "/home/jscesar/Tcc/datasets/lfw2r_train"
validation_path = "/home/jscesar/Tcc/datasets/lfw2r_validation"

train_datagen = ImageDataGenerator(
    horizontal_flip=True, zoom_range=[0.5, 1.0]
).flow_from_directory(
    directory=train_path,
    target_size=IMG_DIM,
    batch_size=batch_size,
    class_mode="categorical",
    interpolation="nearest"
)
val_datagen = ImageDataGenerator(
    horizontal_flip=True, zoom_range=[0.5, 1.0]
).flow_from_directory(
    directory=validation_path,
    target_size=IMG_DIM,
    batch_size=batch_size,
    class_mode="categorical",
    interpolation="nearest"
)

xception, preprocess_input = Classifiers.get('xception')
model = xception(
    include_top=False,
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    weights='imagenet')

output = model.layers[-1].output
output = keras.layers.Flatten()(output)

model = Model(model.input, outputs=output)
model.trainable = True

for layer in model.layers:
    layer.trainable = False

dropout = Dropout(0.3)(model.layers[-1].output)
out = Dense(64, activation='softmax', activity_regularizer=l2(1e-4))(dropout)

new_model = Model(inputs=model.inputs, outputs=out)
new_model.summary()
new_model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=2e-6),
    metrics=['accuracy'])

callback = EarlyStopping(monitor='loss', patience=3)

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=model_path,
    histogram_freq=1,
    write_images=True,
    write_graph=True)

history_1 = new_model.fit_generator(train_datagen,
                                    steps_per_epoch=train_size / batch_size,
                                    epochs=40,
                                    validation_data=val_datagen,
                                    validation_steps=validation_size / batch_size,
                                    verbose=1,
                                    callbacks=[callback, tensorboard_callback])

new_model.save(model_name + ".h5")

