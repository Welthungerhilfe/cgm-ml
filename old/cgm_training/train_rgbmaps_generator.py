'''
This script trains on RGB-Maps.
'''
import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")
from cgmcore import modelutils
from cgmcore import utils
import numpy as np
from tensorflow.keras import callbacks, optimizers, models, layers
import pprint
import os
from cgmcore.preprocesseddatagenerator import get_dataset_path, create_datagenerator_from_parameters
import random
import qrcodes
from cgmcore.utils import create_training_tasks
import multiprocessing

# Get the dataset path.
dataset_path = get_dataset_path()
print("Using dataset path", dataset_path)

# Hyperparameters.
steps_per_epoch = 100
validation_steps = 10
epochs = 100
batch_size = 64
random_seed = 667

image_size = 128

# For creating pointclouds.
dataset_parameters = {}
dataset_parameters["input_type"] = "rgbmap"
dataset_parameters["output_targets"] = ["height"]
dataset_parameters["random_seed"] = 666
dataset_parameters["filter"] = "360"
dataset_parameters["sequence_length"] = 0#4
dataset_parameters["rgbmap_target_width"] = image_size
dataset_parameters["rgbmap_target_height"] = image_size
dataset_parameters["rgbmap_scale_factor"] = 1.0
dataset_parameters["rgbmap_axis"] = "horizontal"
datagenerator_instance = create_datagenerator_from_parameters(dataset_path, dataset_parameters)

# Get the QR-codes.
qrcodes = datagenerator_instance.qrcodes[:]
subset_sizes = [0.25, 0.5, 0.75, 1.0]    
qrcodes_tasks = create_training_tasks(qrcodes, subset_sizes)

# Go through all.
for qrcodes_task in qrcodes_tasks:
    qrcodes_train, qrcodes_validate = qrcodes_task
    print("QR-codes for training:\n", "\t".join(qrcodes_train))
    print("QR-codes for validation:\n", "\t".join(qrcodes_validate))

    # Create python generators.
    generator_train = datagenerator_instance.generate(size=batch_size, qrcodes_to_use=qrcodes_train)
    generator_validate = datagenerator_instance.generate(size=batch_size, qrcodes_to_use=qrcodes_validate)

    # Testing the genrators.
    def test_generator(generator):
        data = next(generator)
        print("Input:", data[0].shape, "Output:", data[1].shape)
    test_generator(generator_train)
    test_generator(generator_validate)

    # Training details.
    training_details = {
        "dataset_path" : dataset_path,
        "qrcodes_train" : qrcodes_train,
        "qrcodes_validate" : qrcodes_validate,
        "steps_per_epoch" : steps_per_epoch,
        "validation_steps" : validation_steps,
        "epochs" : epochs,
        "batch_size" : batch_size,
        "random_seed" : random_seed,
        "dataset_parameters" : dataset_parameters
    }

    # Date time string.
    datetime_string = utils.get_datetime_string() + "_{}-{}".format(len(qrcodes_train), len(qrcodes_validate)) + "_".join(dataset_parameters["output_targets"])

    # Output path. Ensure its existence.
    output_path = os.path.join("/whhdata/models", datetime_string)
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    print("Using output path:", output_path)

    # Important things.
    pp = pprint.PrettyPrinter(indent=4)
    log_dir = os.path.join("/whhdata/models", "logs", datetime_string)
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir)
    histories = {}

    # Training network.
    def train_rgbmaps():

        sequence_length = dataset_parameters["sequence_length"]

        model = models.Sequential()
        if sequence_length == 0:
            model.add(layers.Conv2D(64, (3,3), activation="relu", input_shape=(image_size, image_size, 3)))
        else:
            model.add(layers.Permute((2, 3, 1, 4), input_shape=(sequence_length, image_size, image_size, 3)))
            model.add(layers.Reshape((image_size, image_size, sequence_length * 3)))
            model.add(layers.Conv2D(64, (3,3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3,3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3,3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3,3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(1, activation="linear"))
        model.summary()


        # Compile the model.
        optimizer = "rmsprop"
        #optimizer = optimizers.Adagrad(lr=0.7, epsilon=None, decay=0.2)
        model.compile(
                optimizer=optimizer,
                loss="mse",
                metrics=["mae"]
            )

        # Train the model.
        history = model.fit_generator(
            generator_train,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=generator_validate,
            validation_steps=validation_steps,
            use_multiprocessing=True,
            workers=multiprocessing.cpu_count() - 1,
            callbacks=[tensorboard_callback]
            )

        histories["rgbnet"] = history
        modelutils.save_model_and_history(output_path, datetime_string, model, history, training_details, "rgbnet")

    train_rgbmaps()