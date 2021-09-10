import os
import shutil


path = '/home/jscesar/datasets/muct'
lfw_train = '/home/jscesar/datasets/muct_train'
lfw_validation = '/home/jscesar/datasets/muct_validation'
lfw_test = '/home/jscesar/datasets/lfw_test'
samples = os.listdir(path)

samples_selected = []

if not os.path.exists(lfw_train):
    os.mkdir(lfw_train)

if not os.path.exists(lfw_validation):
    os.mkdir(lfw_validation)

for folder in samples:
    folder_path = os.path.join(path, folder)

    files = os.listdir(folder_path)
    data_folder = os.path.join(lfw_train, folder)
    val_folder = os.path.join(lfw_validation, folder)

    print(folder)
    samples_selected.append(folder)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)
    count = 0
    for file in files:
        if count <= 11:
            shutil.copyfile(os.path.join(folder_path, file), os.path.join(data_folder, file))
        elif count > 11:
            shutil.copyfile(os.path.join(folder_path, file), os.path.join(val_folder, file))
        count = count + 1