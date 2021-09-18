from face_detect import detect_face, rotate_images
import os
import cv2


def extract_face(path):
    samples = os.listdir(path)

    lfw_train = path.replace('muct', 'muct_r')

    if not os.path.exists(lfw_train):
        os.mkdir(lfw_train)

    samples_selected = []
    cc = 0
    for folder in samples:
        folder_path = os.path.join(path, folder)
        new_path = os.path.join(lfw_train, folder)

        if not os.path.exists(new_path):
            os.mkdir(new_path)

        files = os.listdir(folder_path)
        for file in files:
            cc += 1
            #image = detect_face(os.path.join(folder_path, file))
            image_path = os.path.join(new_path, file)
            #cv2.imwrite(image_path, image)

            rotate_images(os.path.join(folder_path, file), image_path)

    print(cc)

path_train = '/home/jscesar/Tcc/datasets/muct_train'
path_validation = '/home/jscesar/Tcc/datasets/muct_validation'
path_test = '/home/jscesar/Tcc/datasets/muct_test'

extract_face(path_train)
extract_face(path_validation)
extract_face(path_test)