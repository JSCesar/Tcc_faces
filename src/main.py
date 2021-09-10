
import numpy as np
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model, load_model
from keras.preprocessing import image

#matplotlib inline
from keras_preprocessing.image import ImageDataGenerator

IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

test_path = '/home/jscesar/datasets/lfw_test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = ImageDataGenerator().flow_from_directory(
    directory=test_path,
    target_size=IMG_DIM,
    batch_size=9,
    interpolation="nearest"
)


#res = model.evaluate(x=test_generator, batch_size=10)

#tt = model.evaluate_generator(test_generator)

path= '/home/jscesar/datasets/lfw_test/Jiang_Zemin/Jiang_Zemin_0017_rotate180.jpg'
model = load_model('modelo.h5')
img = image.load_img(path, target_size=(300, 300))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)
prediction = model.predict(img_preprocessed)
label_names = list(test_generator.class_indices.keys())
labels_index = list(test_generator.class_indices.values())
print(label_names[prediction.argmax()])
