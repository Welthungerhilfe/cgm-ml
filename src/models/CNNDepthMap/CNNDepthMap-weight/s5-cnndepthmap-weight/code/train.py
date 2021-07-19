import os
import pickle
import random
import logging
import logging.config

from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.run import Run
import glob2 as glob
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks

from preprocessing import preprocess_depthmap, preprocess_targets

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

# Get the current run.
run = Run.get_context()

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if(run.id.startswith("OfflineRun")):
    logging.info('Running in offline mode...')

    # Access workspace.
    logging.info('Accessing workspace...')
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "s5-cnndepthmap-weight-offline")
    run = experiment.start_logging(outputs=None, snapshot_directory=".")

    # Get dataset.
    logging.info('Accessing dataset...')
    if not os.path.exists("premiumfileshare"):
        assert False, "Requires small size dataset"
        dataset_name = "cgmmldevpremium-SampleDataset-Example"
        dataset = workspace.datasets[dataset_name]
        dataset.download(target_path='.', overwrite=False)
    dataset_path = glob.glob(os.path.join("premiumfileshare"))[0]

# Online run. Use dataset provided by training notebook.
else:
    logging.info('Running in online mode...')
    experiment = run.experiment
    workspace = experiment.workspace
    dataset_path = run.input_datasets["dataset"]

# Get the QR-code paths.
logging.info('Dataset path: %s', dataset_path)
logging.info(glob.glob(os.path.join(dataset_path, "*")))  # Debug
logging.info('Getting QR-code paths...')
qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))

# Shuffle and split into train and validate.
random.shuffle(qrcode_paths)
split_index = int(len(qrcode_paths) * 0.8)
qrcode_paths_training = qrcode_paths[:split_index]
qrcode_paths_validate = qrcode_paths[split_index:]
del qrcode_paths

# Show split.
logging.info('Paths for training: \n\t' + '\n\t'.join(qrcode_paths_training))
logging.info('Paths for validation: \n\t' + '\n\t'.join(qrcode_paths_validate))

assert len(qrcode_paths_training) > 0 and len(qrcode_paths_validate) > 0


def get_depthmap_files(paths):
    pickle_paths = []
    for path in paths:
        pickle_paths.extend(glob.glob(os.path.join(path, "**", "*.p")))
    return pickle_paths


# Get the pointclouds.
logging.info('Getting depthmap paths...')
paths_training = get_depthmap_files(qrcode_paths_training)
paths_validate = get_depthmap_files(qrcode_paths_validate)
del qrcode_paths_training
del qrcode_paths_validate
logging.info('Using %d files for training.', len(paths_training))
logging.info('Using %d files for validation.', len(paths_validate))

image_target_height = 224
image_target_width = 172

# Function for loading and processing depthmaps.


def tf_load_pickle(path):

    def py_load_pickle(path):
        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = preprocess_depthmap(depthmap)
        targets = preprocess_targets(targets, targets_indices)
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path], [tf.float32, tf.float32])
    depthmap.set_shape((image_target_height, image_target_width, 4))
    targets.set_shape((len(targets_indices,)))
    return depthmap, targets


def tf_flip(image):

    image = tf.image.random_flip_left_right(image)
    return image


# Parameters for dataset generation.
shuffle_buffer_size = 64
subsample_size = 1024
channels = list(range(0, 3))
targets_indices = [1]  # 0 is height, 1 is weight.

# Create dataset for training.
paths = paths_training
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(lambda path: tf_load_pickle(path))
dataset = dataset.cache()
dataset = dataset.shuffle(shuffle_buffer_size)
dataset = dataset.map(lambda image, label: (tf_flip(image), label))
#dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset_training = dataset
del dataset

# Create dataset for validation.
# Note: No shuffle necessary.
paths = paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(lambda path: tf_load_pickle(path))
dataset = dataset.cache()
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validate = dataset
del dataset


# Note: Now the datasets are prepared.

# Instantiate model.
model = models.Sequential()

model.add(
    layers.Conv2D(
        filters=8,
        kernel_size=(
            3,
            3),
        padding="same",
        activation="relu",
        input_shape=(
                image_target_height,
                image_target_width,
            4)))
model.add(layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1, activation="linear"))
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

# Add checkpoint callback.
best_model_path = "best_model.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
training_callbacks.append(checkpoint_callback)

# Compile the model.
model.compile(
    optimizer="nadam",
    loss="mse",
    metrics=["mae"]
)

batch_size = 256
epochs = 8000
model.fit(
    dataset_training.batch(batch_size),
    validation_data=dataset_validate.batch(batch_size),
    epochs=epochs,
    callbacks=training_callbacks
)

run.upload_file(name=best_model_path, path_or_stream=best_model_path)

# Save the weights.
#logging.info("Saving and uploading weights...")
#path = "cnndepthmap_weights.h5"
#model.save_weights(path)
#run.upload_file(name="cnndepthmap_weights.h5", path_or_stream=path)

# Save the model.
#logging.info("Saving and uploading model...")
#path = "cnndepthmap_model"
#model.save(path)
#run.upload_folder(name="cnndepthmap", path=path)


# Done.
run.complete()
