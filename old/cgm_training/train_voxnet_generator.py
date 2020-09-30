'''
This script trains VoxNet.
'''
from cgmcore import modelutils
from cgmcore import utils
import numpy as np
from keras import callbacks
import pprint
import os
from cgmcore.etldatagenerator import get_dataset_path, create_datagenerator_from_parameters
import random

# Get the dataset path.
dataset_path = get_dataset_path()
print("Using dataset path", dataset_path)

# Hyperparameters.
steps_per_epoch = 100
validation_steps = 10
epochs = 100
batch_size = 32
random_seed = 667

# For creating voxelgrids.
dataset_parameters_voxelgrids = {}
dataset_parameters_voxelgrids["input_type"] = "voxelgrid"
dataset_parameters_voxelgrids["output_targets"] = ["height"]
dataset_parameters_voxelgrids["random_seed"] = random_seed
dataset_parameters_voxelgrids["voxelgrid_target_shape"] = (32, 32, 32)
dataset_parameters_voxelgrids["voxel_size_meters"] = 0.1
dataset_parameters_voxelgrids["voxelgrid_random_rotation"] = False
dataset_parameters_voxelgrids["sequence_length"] = 0
datagenerator_instance_voxelgrids = create_datagenerator_from_parameters(dataset_path, dataset_parameters_voxelgrids)

# Do the split.
random.seed(random_seed)
qrcodes_shuffle = datagenerator_instance_voxelgrids.qrcodes[:]
random.shuffle(qrcodes_shuffle)
split_index = int(0.8 * len(qrcodes_shuffle))
qrcodes_train = sorted(qrcodes_shuffle[:split_index])
qrcodes_validate = sorted(qrcodes_shuffle[split_index:])
del qrcodes_shuffle
print("QR-codes for training:\n", "\t".join(qrcodes_train))
print("QR-codes for validation:\n", "\t".join(qrcodes_validate))

# Create python generators.
generator_voxelgrids_train = datagenerator_instance_voxelgrids.generate(size=batch_size, qrcodes_to_use=qrcodes_train)
generator_voxelgrids_validate = datagenerator_instance_voxelgrids.generate(size=batch_size, qrcodes_to_use=qrcodes_validate)

# Testing the generators.
def test_generator(generator):
    data = next(generator)
    print("Input:", data[0].shape, "Output:", data[1].shape)
#test_generator(generator_voxelgrids_train)
#test_generator(generator_voxelgrids_validate)

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
}

# Output path. Ensure its existence.
output_path = "models"
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
print("Using output path:", output_path)

# Important things.
pp = pprint.PrettyPrinter(indent=4)
tensorboard_callback = callbacks.TensorBoard()

# Training VoxNet.
def train_voxnet():
    print("Training VoxNet...")

    # Create the model.
    input_shape = (32, 32, 32)
    output_size = 1
    model_voxnet = modelutils.create_voxnet_model_homepage(input_shape, output_size)
    model_voxnet.summary()

    # Compile the model.
    model_voxnet.compile(
            optimizer="rmsprop",
            loss="mse",
            metrics=["mae"]
        )

    # Train the model.
    history = model_voxnet.fit_generator(
        generator_voxelgrids_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=generator_voxelgrids_validate,
        validation_steps=validation_steps,
        callbacks=[tensorboard_callback]
        )

    modelutils.save_model_and_history(output_path, model_voxnet, history, training_details, "voxnet")

train_voxnet()