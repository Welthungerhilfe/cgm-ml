import sys
sys.path.insert(0, "..")
from cgm_database import dbutils
import os
import random
import numpy as np
import shutil
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

main_connector = dbutils.connect_to_main_database()

def get_artifact_paths(artifact_status, artifact_type):
    # Query database.
    sql_statement = ""
    sql_statement += "SELECT a.path FROM measure_quality mq"
    sql_statement += " INNER JOIN artifact a ON mq.measure_id = a.measure_id"
    sql_statement += " WHERE mq.key='expert_status' AND mq.text_value='{}'".format(artifact_status)
    sql_statement += " AND a.type='{}'".format(artifact_type)
    sql_statement += ";"
    results = main_connector.execute(sql_statement, fetch_all=True)
    results = [x[0].replace("whhdata", "localssd") for x in results]
    return results
    

# Paths.
dataset_path = "/localssd/standing_lying_data"
dataset_path_train = os.path.join(dataset_path, "train")
dataset_path_validate = os.path.join(dataset_path, "validate")
dataset_path_test = os.path.join(dataset_path, "test")

def copy_files(source_paths, destination_path):
    for source_path in tqdm(source_paths):
        shutil.copy(source_path, destination_path)

if os.path.exists(dataset_path) != True:
    
    # Create folders.
    os.mkdir(dataset_path)
    for path in [dataset_path_train, dataset_path_validate, dataset_path_test]:
        os.mkdir(path)
        os.mkdir(os.path.join(path, "standing"))
        os.mkdir(os.path.join(path, "lying"))
        
    # Get the image paths from database.
    image_paths_standing = get_artifact_paths("standing", "rgb")
    image_paths_lying = get_artifact_paths("lying", "rgb")

    # Shuffle dataset.
    random.shuffle(image_paths_standing)
    random.shuffle(image_paths_lying)

    # Ensure equal length. Balancing dataset.
    maximum_length = min(len(image_paths_standing), len(image_paths_lying))
    image_paths_standing = image_paths_standing[:maximum_length]
    image_paths_lying = image_paths_lying[:maximum_length]
    
    # Get indices for split.
    split_ratio = (7, 1, 2)
    split_ratio_sum = np.sum(split_ratio)
    end_index_train = int(maximum_length * split_ratio[0] / split_ratio_sum)
    end_index_validate = int(maximum_length * (split_ratio[0] + split_ratio[1]) / split_ratio_sum)
    
    # Copy train.
    print("Copying files for train...")
    copy_files(image_paths_standing[0:end_index_train], os.path.join(dataset_path_train, "standing"))
    copy_files(image_paths_lying[0:end_index_train], os.path.join(dataset_path_train, "lying"))
    print("Done.")

    # Copy validate.
    print("Copying files for validate...")
    copy_files(image_paths_standing[end_index_train:end_index_validate], os.path.join(dataset_path_validate, "standing"))
    copy_files(image_paths_lying[end_index_train:end_index_validate], os.path.join(dataset_path_validate, "lying"))
    print("Done.")

    # Copy test.
    print("Copying files for test...")
    copy_files(image_paths_standing[end_index_validate:], os.path.join(dataset_path_test, "standing"))
    copy_files(image_paths_lying[end_index_validate:], os.path.join(dataset_path_test, "lying"))
    print("Done.")

else:
    print("Dataset path exists. Not doing anyting. Delete it manually and restart if you like.")


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        dataset_path_train,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary")

validation_generator = test_datagen.flow_from_directory(
        dataset_path_validate,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary")

from tensorflow.keras import applications, models, layers, optimizers

vgg19 = applications.VGG19(
    weights="imagenet", 
    include_top=False, 
    input_shape=(150, 150, 3)
)

vgg19.summary()

vgg19.trainable = False

model = models.Sequential()
model.add(vgg19)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(
    optimizer=optimizers.RMSprop(lr=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)

model.save("model.h5")

import matplotlib.pyplot as plt

plt.plot(history.history["acc"], label="acc")
plt.plot(history.history["val_acc"], label="val_acc")
plt.legend()
plt.savefig("model.png")
plt.close()