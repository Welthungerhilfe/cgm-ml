import azureml
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.run import Run
import os
import logging
import logging.config
import glob2 as glob
from gapnet.models import GAPNet
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
import numpy as np
import pickle
import random
from preprocessing import preprocess_pointcloud, preprocess_targets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

# Get the current run.
run = Run.get_context()

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if(run.id.startswith("OfflineRun")):
    logging.info('Running in offline mode...')

    # Access workspace.
    logging.info('Accessing workspace...')
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "gapnet-offline")
    run = experiment.start_logging(outputs=None, snapshot_directory=".")

    # Get dataset.
    logging.info('Accessing dataset...')
    if os.path.exists("premiumfileshare") == False:
        dataset_name = "cgmmldevpremium-SampleDataset-Example"
        dataset = workspace.datasets[dataset_name]
        dataset.download(target_path='.', overwrite=False)
    dataset_path = glob.glob(os.path.join("premiumfileshare", "*"))[0]

# Online run. Use dataset provided by training notebook.
else:
    logging.info('Running in online mode...')
    experiment = run.experiment
    workspace = experiment.workspace
    dataset_path = run.input_datasets["dataset"]

# Get the QR-code paths.
logging.info('Dataset path: %s', dataset_path)
logging.info('Getting QR-code paths...')
qrcode_paths = glob.glob(os.path.join(dataset_path, "pcd", "*",))

# Shuffle and split into train and validate.
random.shuffle(qrcode_paths)
split_index = int(len(qrcode_paths) * 0.8)
qrcode_paths_training = qrcode_paths[:split_index]
qrcode_paths_validate = qrcode_paths[split_index:]
del qrcode_paths

# Show split.
logging.info('Paths for training: \n\t' + '\n\t'.join(qrcode_paths_training))
logging.info('Paths for validation: \n\t' + '\n\t'.join(qrcode_paths_validate))


def get_pickle_files(paths):
    pickle_paths = []
    for path in paths:
        pickle_paths.extend(glob.glob(os.path.join(path, "**", "*.p")))
    return pickle_paths


# Get the pointclouds.
logging.info('Getting pointcloud paths...')
paths_training = get_pickle_files(qrcode_paths_training)
paths_validate = get_pickle_files(qrcode_paths_validate)
del qrcode_paths_training
del qrcode_paths_validate
logging.info('Using %d files for training.', len(paths_training))
logging.info('Using %d files for validation.', len(paths_validate))


# Function for loading and subsampling pointclouds.


def tf_load_pickle(path, subsample_size, channels, targets_indices):
    assert isinstance(channels, list)
    assert isinstance(targets_indices, list)

    def py_load_pickle(path):
        pointcloud, targets = pickle.load(open(path.numpy(), "rb"))
        pointcloud = preprocess_pointcloud(pointcloud, subsample_size, channels)
        targets = preprocess_targets(targets, targets_indices)
        return pointcloud, targets

    pointcloud, targets = tf.py_function(py_load_pickle, [path], [tf.float32, tf.float32])
    pointcloud.set_shape((subsample_size, len(channels)))
    targets.set_shape((len(targets_indices,)))
    return pointcloud, targets


# Parameters for dataset generation.
shuffle_buffer_size = 64
subsample_size = 1024
channels = list(range(0, 3))
targets_indices = [0]  # 0 is height, 1 is weight.

# Create dataset for training.
paths = paths_training
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(lambda path: tf_load_pickle(path, subsample_size, channels, targets_indices))
dataset = dataset.cache()
dataset = dataset.shuffle(shuffle_buffer_size)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset_training = dataset
del dataset

# Create dataset for validation.
# Note: No shuffle necessary.
paths = paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(lambda path: tf_load_pickle(path, subsample_size, channels, targets_indices))
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validate = dataset
del dataset


# Note: Now the datasets are prepared.


# Start logging.
#run = experiment.start_logging()

# Intantiate GAPNet.
model = GAPNet()
model.summary()

# Get ready to add callbacks.
training_callbacks = []

# Pushes metrics and losses into the run on AzureML.


class AzureLogCallback(callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                run.log(key, value)


training_callbacks.append(AzureLogCallback())

# Add TensorBoard callback.
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs",
    histogram_freq=0,
    write_graph=True,
    write_grads=False,
    write_images=True,
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None,
    embeddings_data=None,
    update_freq="epoch"
)
training_callbacks.append(tensorboard_callback)


# Compile the model.
lr = 0.0001
adam = optimizers.Adam(lr=lr)
model.compile(
    optimizer=adam,
    loss="mse",
    metrics=["mae"]
)

batch_size = 128
epochs = 500
model.fit(
    dataset_training.batch(batch_size),
    validation_data=dataset_validate.batch(batch_size),
    epochs=epochs,
    callbacks=training_callbacks
)

# Save the model.
logging.info('Saving and uploading weights...')
path = "gapnet_weights.h5"
model.save_weights(path)
run.upload_file(name="gapnet_weights.h5", path_or_stream=path)

# Save the model.
logging.info('Saving and uploading model...')
path = "gapnet_model"
model.save(path)
run.upload_folder(name="gapnet", path=path)


# Done.
run.complete()
