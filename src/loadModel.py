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
    train_datagen = ImageDataGenerator(
        horizontal_flip=True
        ).flow_from_directory(
            directory="/home/jscesar/Tcc/datasets/lfw_faces_train",
            target_size=(300,300),
            batch_size=16,
            class_mode="categorical",
            interpolation="nearest"
    )
    labels = (train_datagen.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    model = load_model('resnext_2021_09_09_12_57_23_AM.h5')

    orig = cv2.imread(image_path)
    resized = cv2.resize(orig, (IMG_WIDTH, IMG_HEIGHT))

    image = load_img(image_path, target_size=(IMG_WIDTH,IMG_HEIGHT))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    preds = model.predict(image)

    i = np.argmax(preds[0])

    label = "{}: {:.2f}%".format(labels[i], preds[0][i]*100)
    print(label)
    print(i)

    cam = GradCam(model, i)
    heatmap = cam.compute_heatmap(image)

    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
               0.4, (255, 255, 255), 2)

    output = np.vstack([orig, heatmap, output])
    output = imutils.resize(output, height=600)
    cv2.imshow("Output", output)
    cv2.waitKey(0)


#montar_gradcam("/home/jscesar/Tcc/datasets/lfw_faces_test/Angelina_Jolie/Angelina_Jolie_0015.jpg")



'''IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
batch_size=32

train_path = "/home/jscesar/datasets/lfw_train"
validation_path="/home/jscesar/datasets/lfw_validation"

test = ImageDataGenerator(zoom_range=[0.5,1.0]).flow_from_directory(
    directory=validation_path,
    target_size=IMG_DIM,
    batch_size=batch_size,
    class_mode="categorical",
    interpolation="nearest",

)

model = load_model('modeloResnext3.h5')
model.evaluate(test)'''
