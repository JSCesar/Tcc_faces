import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator, array_to_img
from tensorflow.keras.applications import imagenet_utils
from face_detect import detect_face
import imutils
from gradcam import GradCam

IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
batch_size=16

test_path="/home/jscesar/Tcc/datasets/muct_r_test"

test = ImageDataGenerator(
    horizontal_flip=True, zoom_range=[0.5, 1.0]
).flow_from_directory(
    directory=test_path,
    target_size=IMG_DIM,
    batch_size=batch_size,
    class_mode="categorical",
    interpolation="nearest"
)

model = load_model('./muct_resnext_2021_09_10_11_55_07_PM.h5')
model.evaluate(test)
