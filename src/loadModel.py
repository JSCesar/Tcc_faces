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

def montar_gradcam(image_path):
    IMG_WIDTH = 300
    IMG_HEIGHT = 300

    orig = cv2.imread(image_path)
    orig = cv2.resize(orig, (IMG_WIDTH, IMG_HEIGHT))

    image = load_img(image_path, target_size=(IMG_WIDTH,IMG_HEIGHT))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    i = np.argmax(preds[0])

    label = "{}: {:.2f}%".format(labels[i], preds[0][i]*100)
    cam = GradCam(model, i)
    heatmap = cam.compute_heatmap(image)

    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    cv2.rectangle(output, (0, 0), (600, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
               1.2, (255, 255, 255), 2)

    output = np.vstack([orig, heatmap, output])
    output = imutils.resize(output, height=600)
    return output


def predict_dataset(dataset_path):
    for folder in os.listdir(dataset_path):
        expected = folder
        folder_path = os.path.join(dataset_path, folder)
        print(expected + "\n")
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            image = load_img(file_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            preds = model.predict(image)
            i = np.argmax(preds[0])
            print("\t" + labels[i])


dataset_path = "/home/jscesar/Tcc/datasets/muct_r_test"
sample = "Julio/Julio_03.jpg"
train_datagen = ImageDataGenerator().flow_from_directory(
            directory=dataset_path,
            target_size=(IMG_WIDTH,IMG_HEIGHT),
            batch_size=16,
            class_mode="categorical",
            interpolation="nearest"
    )

labels = (train_datagen.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(labels)

model = load_model("./muct_resnext_2021_09_10_11_55_07_PM.h5")

gradcam = montar_gradcam("/home/jscesar/Tcc/datasets/muct_r_test/Julio/Julio_03.jpeg")

cv2.imshow("output", gradcam)
cv2.waitKey(0)





