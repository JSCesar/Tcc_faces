import os
import shutil


path = '/home/jscesar/Tcc/datasets/muct'
train = "/home/jscesar/Tcc/datasets/muct_train"
validation = '/home/jscesar/Tcc/datasets/muct_validation'
test = '/home/jscesar/Tcc/datasets/muct_test'
samples = os.listdir(path)

samples_selected = []

if not os.path.exists(train):
    os.mkdir(train)

if not os.path.exists(validation):
    os.mkdir(validation)

if not os.path.exists(test):
    os.mkdir(test)

for folder in samples:
    folder_path = os.path.join(path, folder)
    files = os.listdir(folder_path)
    data_folder = os.path.join(train, folder)
    val_folder = os.path.join(validation, folder)
    test_folder = os.path.join(test, folder)

    if len(files) >= 15:
        print(folder)
        samples_selected.append(folder)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)
        if not os.path.exists(val_folder):
            os.mkdir(val_folder)
    num = 0
    for file in files:
        if len(files) >= 15:

            num = num + 1
            if num <= 9:
                shutil.copyfile(os.path.join(folder_path, file), os.path.join(data_folder, file))
            if num > 9 and num <= 12:
                shutil.copyfile(os.path.join(folder_path, file), os.path.join(test_folder, file))
            elif num > 12 and num <= 15:
                shutil.copyfile(os.path.join(folder_path, file), os.path.join(val_folder, file))