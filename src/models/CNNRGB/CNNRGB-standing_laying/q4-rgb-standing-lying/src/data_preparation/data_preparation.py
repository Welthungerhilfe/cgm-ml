# Mount the container with the compute instance

import pandas as pd
import random
from shutil import copyfile
from PIL import Image
import os

#  Parent directory
PARENT_DIRECTORY = '/mnt/preprocess/'
BASE_DIRECTORY_PATH = os.path.join(PARENT_DIRECTORY, 'standing_laying_v1.0/')
if not os.path.exists(BASE_DIRECTORY_PATH):
    os.mkdir(BASE_DIRECTORY_PATH)

#  Train directory
TRAIN_DIRECTORY_PATH = os.path.join(BASE_DIRECTORY_PATH, 'train/')
if not os.path.exists(TRAIN_DIRECTORY_PATH):
    os.mkdir(TRAIN_DIRECTORY_PATH)

#  Standing directory
STANDING_PATH = os.path.join(TRAIN_DIRECTORY_PATH, 'standing/')
if not os.path.exists(STANDING_PATH):
    os.mkdir(STANDING_PATH)

#  laying directory
LAYINGING_PATH = os.path.join(TRAIN_DIRECTORY_PATH, 'laying/')
if not os.path.exists(LAYINGING_PATH):
    os.mkdir(LAYINGING_PATH)


#  Read CSV from the ML Database
#  To generate CSV use 'select_artifacts_with_targets' view with change in Dataformat(jpg)

artifacts = pd.read_csv('training_data.csv')
print("Shape of artifacts in ML database" + artifacts.shape)

#  To get only good artifacts
artifacts = artifacts[artifacts['tag'] == 'good']
print("Shape of artifacts tagged good" + artifacts.shape)

artifacts.head()

del artifacts['tag']
del artifacts['scan_group']
del artifacts['id']

#  label the Data between standing and laying
#  1 is standing
#  0 is laying
artifacts['label'] = artifacts.apply(lambda x: 1 if int(int(x['key']) / 100) == 1 else 0, axis=1)

#  Print the value count in artifacts level
print(artifacts['label'].value_counts())

#################################################################
#  This function copy the artfacts from one folder to another  ##
#  and Rotatting the laying child by 90 Anti-clockwise and     ##
#  Standing child by 270 Anti-clockwise                        ##


def find_artifacts_by_label(label, start_point, end_point, artifacts):
    df = artifacts[artifacts['label'] == label]
    df_paths = df['storage_path'].values.tolist()
    random.shuffle(df_paths)
    file_names = []
    for path in df_paths:
        p = path.split('/')
        file_names.append(p[5])
    print(len(file_names))
    standing_path = df_paths[start_point:end_point]
    file_name = file_names[start_point:end_point]
    for (i, j) in zip(standing_path, file_name):
        src = PARENT_DIRECTORY + 'cgminbmzprod_v5.0/'
        src += i
        if label == 0:
            dst = PARENT_DIRECTORY + 'standing_laying_v1.0/train/laying/'
            dst += j
            copyfile(src, dst)
            im = Image.open(dst)
            im = im.rotate(90, expand=True)
            im.show()
            im.save(dst)
        elif label == 1:
            dst = PARENT_DIRECTORY + 'standing_laying_v1.0/train/standing/'
            dst += j
            copyfile(src, dst)
            im = Image.open(dst)
            im = im.rotate(90, expand=True)
            im.show()
            im.save(dst)

#######################################################################


# 8000 data process
find_artifacts_by_label(label=0, start_point=0, end_point=8000, artifacts=artifacts)
find_artifacts_by_label(label=1, start_point=0, end_point=8000, artifacts=artifacts)
