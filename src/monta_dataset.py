import os
import shutil


path = '/home/jscesar/lfw'
lfw_train = "/home/jscesar/Tcc/datasets/lfw2_train"
lfw_validation = '/home/jscesar/Tcc/datasets/lfw2_validation'
lfw_test = '/home/jscesar/Tcc/datasets/lfw2_test'
samples = os.listdir(path)

samples_selected = []

if not os.path.exists(lfw_train):
    os.mkdir(lfw_train)

if not os.path.exists(lfw_validation):
    os.mkdir(lfw_validation)

if not os.path.exists(lfw_test):
    os.mkdir(lfw_test)

for folder in samples:
    folder_path = os.path.join(path, folder)
    files = os.listdir(folder_path)
    data_folder = os.path.join(lfw_train, folder)
    val_folder = os.path.join(lfw_validation, folder)
    test_folder = os.path.join(lfw_test, folder)

    if len(files) >= 20:
        print(folder)
        samples_selected.append(folder)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)
        if not os.path.exists(val_folder):
            os.mkdir(val_folder)

    for file in files:
        if len(files) >= 20:
            num = int(file.split('.')[0].split('_')[-1])
            if num <= 12:
                shutil.copyfile(os.path.join(folder_path, file), os.path.join(data_folder, file))
            if num > 12 and num <=16:
                shutil.copyfile(os.path.join(folder_path, file), os.path.join(test_folder, file))
            elif num > 16 and num <= 20:
                shutil.copyfile(os.path.join(folder_path, file), os.path.join(val_folder, file))