import os
import shutil

dataset_path = '/home/jscesar/datasets/muct'

muct_path = '/home/jscesar/datasets/muct'

for file in sorted(os.listdir(muct_path)):
    folder_name = file[0:5]

    if not os.path.exists(os.path.join(dataset_path, folder_name)):
        os.mkdir(os.path.join(dataset_path, folder_name))

    shutil.move(os.path.join(muct_path, file), os.path.join(dataset_path, folder_name))
