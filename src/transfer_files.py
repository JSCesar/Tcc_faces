import os
import shutil
path = '/home/jscesar/datasets/lfw_train'
samples = os.listdir(path)
count = 0

for folder in samples:
    folder_path = os.path.join(path, folder)
    files = os.listdir(folder_path)
    for file in files:
       count = count + 1

print(count)