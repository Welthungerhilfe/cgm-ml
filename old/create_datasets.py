'''
Creates a preprocessed dataset with the OLD (!) datagenerator. This means that the store- and db-data from the mobile apps are processed directly.

Important note: This sourcecode is only here for historical purposes. Once everything is stable again it will be archived.
'''

# Dataset-Generator.
import os
from cgmcore.datagenerator import DataGenerator, get_dataset_path, create_datagenerator_from_parameters
from cgmcore import utils
import pickle
import random
import sys
import multiprocessing

# Prepare for multiprocessing. Determine number of parallel jobs.
multiprocessing_jobs = multiprocessing.cpu_count()
print("Going to spawn", multiprocessing_jobs, "jobs...")

# Get the dataset path.
dataset_path = get_dataset_path()
print("Using dataset path:", dataset_path)

# Output path. Ensure its existence.
output_path = "datasets"
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
print("Using output path:", output_path)

# Now come the parameters for dataset generation.

# For creating pointclouds.
dataset_parameters_pointclouds = {}
dataset_parameters_pointclouds["input_type"] = "pointcloud"
dataset_parameters_pointclouds["output_targets"] = ["height"]
dataset_parameters_pointclouds["random_seed"] = 666
dataset_parameters_pointclouds["pointcloud_target_size"] = 30000
dataset_parameters_pointclouds["pointcloud_random_rotation"] = True
dataset_parameters_pointclouds["dataset_size_train"] = 1000
dataset_parameters_pointclouds["dataset_size_test"] = 200
dataset_parameters_pointclouds["sequence_length"] = 4

# For creating voxelgrids.
dataset_parameters_voxelgrids = {}
dataset_parameters_voxelgrids["input_type"] = "voxelgrid"
dataset_parameters_voxelgrids["output_targets"] = ["height"]
dataset_parameters_voxelgrids["random_seed"] = 666
dataset_parameters_voxelgrids["voxelgrid_target_shape"] = (32, 32, 32)
dataset_parameters_voxelgrids["voxel_size_meters"] = 0.1
dataset_parameters_voxelgrids["voxelgrid_random_rotation"] = True
dataset_parameters_voxelgrids["dataset_size_train"] = 6000 // 4
dataset_parameters_voxelgrids["dataset_size_test"] = 1000 // 4
dataset_parameters_voxelgrids["sequence_length"] = 4

# Define which parameters to use.
dataset_parameters_to_use = []
dataset_parameters_to_use.append(dataset_parameters_pointclouds)
dataset_parameters_to_use.append(dataset_parameters_voxelgrids)

# Analysis.
#analyze = True
analyze = False
if analyze:
    datagenerator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_to_use[0])

    datagenerator.analyze_files()
    datagenerator.analyze_pointclouds()
    datagenerator.analyze_voxelgrids()

# Method for doing the train-test-split and generation.
def split_and_generate(datagenerator, dataset_parameters):

    # Do the split.
    random.seed(dataset_parameters["random_seed"])
    qrcodes_shuffle = datagenerator.qrcodes[:]
    random.shuffle(qrcodes_shuffle)
    split_index = int(0.8 * len(qrcodes_shuffle))
    qrcodes_train = sorted(qrcodes_shuffle[:split_index])
    qrcodes_test = sorted(qrcodes_shuffle[split_index:])
    del qrcodes_shuffle
    print("")

    print("QR-Codes for training:", " ".join(qrcodes_train))
    print("")
    print("QR-Codes for testing:", " ".join(qrcodes_test))
    print("")

    print("Generating training data...")
    dataset_train = next(datagenerator.generate(size=dataset_parameters["dataset_size_train"], qrcodes_to_use=qrcodes_train, yield_file_paths=True, verbose=True, multiprocessing_jobs=multiprocessing_jobs))

    print("Generating testing data...")
    dataset_test = next(datagenerator.generate(size=dataset_parameters["dataset_size_test"], qrcodes_to_use=qrcodes_test, yield_file_paths=True, verbose=True, multiprocessing_jobs=multiprocessing_jobs))

    print("Done.")
    return dataset_train, dataset_test

# Method for saving dataset.
def save_dataset(dataset_train, dataset_test, dataset_parameters):
    print("Saving dataset...")
    data = (dataset_train, dataset_test, dataset_parameters)
    datetime_string = utils.get_datetime_string()
    dataset_name = datetime_string + "-" + dataset_parameters["input_type"] + "-dataset.p"
    dataset_path = os.path.join(output_path, dataset_name)
    pickle.dump(data, open(dataset_path, "wb"))
    print("Saved dataset to " + dataset_path + ".")

# Generate with parameters.
for dataset_parameters in dataset_parameters_to_use:
    print(dataset_parameters["input_type"])
    datagenerator = create_datagenerator_from_parameters(dataset_path, dataset_parameters)
    dataset_train, dataset_test = split_and_generate(datagenerator, dataset_parameters)
    save_dataset(dataset_train, dataset_test, dataset_parameters)
