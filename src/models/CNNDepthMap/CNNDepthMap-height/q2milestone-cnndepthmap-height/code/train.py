import azureml
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.run import Run
import os
import logging
import logging.config
import glob2 as glob
import tensorflow as tf
import random
from tensorflow.keras import models, layers, callbacks, optimizers
import numpy as np
import pickle
import random
from preprocessing import preprocess_depthmap, preprocess_targets
import argparse
from utils import GradCAM, make_grid
import cv2
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

# Parse command line arguments.
parser = argparse.ArgumentParser(description="Training script.")
parser.add_argument('--split_seed', nargs=1, default=0, type=int, help="The random seed for splitting.")
parser.add_argument('--target_size', nargs=1, default="180x240", type=str,
                    help="The target image size format WIDTHxHEIGHT.")
parser.add_argument('--epochs', nargs=1, default=1000, type=int, help="The number of epochs.")
parser.add_argument('--batch_size', nargs=1, default=256, type=int, help="The batch size for training.")
parser.add_argument('--comment', nargs=1, default="No comment.", type=str, help="A comment.")

args = parser.parse_args()

# Get the split seed.
split_seed = args.split_seed[0]

# Get the image target size.
target_size = args.target_size[0].split("x")
image_target_width = int(target_size[0])
image_target_height = int(target_size[1])

# Get batch size and epochs
batch_size = args.batch_size[0]
epochs = args.epochs[0]
batch_size = int(batch_size)
epochs = int(epochs)

# Get the current run.
run = Run.get_context()

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if(run.id.startswith("OfflineRun")):
    logging.info('Running in offline mode...')

    # Access workspace.
    logging.info('Accessing workspace...')
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "training-junkyard")
    run = experiment.start_logging(outputs=None, snapshot_directory=".")

    # Get dataset.
    logging.info('Accessing dataset...')
    if os.path.exists("dataset") == False:
        dataset_name = "anon-depthmaps-56k"
        dataset = workspace.datasets[dataset_name]
        dataset.download(target_path='dataset', overwrite=False)
    dataset_path = "dataset"

# Online run. Use dataset provided by training notebook.
else:
    logging.info('Running in online mode...')
    experiment = run.experiment
    workspace = experiment.workspace
    dataset_path = run.input_datasets["dataset"]

# Get the QR-code paths.
logging.info('Dataset path: %s', dataset_path)
dataset_path = os.path.join(dataset_path, "qrcode")
logging.info(glob.glob(os.path.join(dataset_path, "*")))  # Debug
logging.info('Getting QR-code paths...')
qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
logging.info('qrcode_paths: %d', len(qrcode_paths))

# Shuffle and split into train and validate.
random.seed(split_seed)
random.shuffle(qrcode_paths)
split_index = int(len(qrcode_paths) * 0.8)
qrcode_paths_training = qrcode_paths[:split_index]
qrcode_paths_validate = qrcode_paths[split_index:]
qrcode_paths_activation = random.choice(qrcode_paths_validate)
qrcode_paths_activation = [qrcode_paths_activation]

del qrcode_paths

# Show split.
logging.info('Paths for training: \n\t' + '\n\t'.join(qrcode_paths_training))
logging.info('Paths for validation: \n\t' + '\n\t'.join(qrcode_paths_validate))
logging.info('Paths for activation: \n\t' + '\n\t'.join(qrcode_paths_activation))

logging.info('Nbr of qrcode_paths for training: %d', len(qrcode_paths_training))
logging.info('Nbr of qrcode_paths for validation: %d', len(qrcode_paths_validate))

assert len(qrcode_paths_training) > 0 and len(qrcode_paths_validate) > 0


def get_depthmap_files(paths):
    pickle_paths = []
    for path in paths:
        pickle_paths.extend(glob.glob(os.path.join(path, "*.p")))
    return pickle_paths


# Get the pointclouds.
logging.info('Getting depthmap paths...')
paths_training = get_depthmap_files(qrcode_paths_training)
paths_validate = get_depthmap_files(qrcode_paths_validate)
paths_activate = get_depthmap_files(qrcode_paths_activation)

del qrcode_paths_training
del qrcode_paths_validate
del qrcode_paths_activation

logging.info('Using %d files for training.', len(paths_training))
logging.info('Using %d files for validation.', len(paths_validate))
logging.info('Using %d files for activation.', len(paths_activate))

def calculate_pickle(path):
    def find_max(path):
        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = preprocess_depthmap(depthmap)
        depthmax = tf.math.reduce_max(depthmap)
        return depthmax
    sample = []
    depthmap_maximum = tf.py_function(find_max, [path], [tf.float32])
    sample.append(depthmap_maximum)
    return sample


def get_max(dataset):
    global_max = []
    for value in dataset:
        max_value = value[0].numpy()[0]
        global_max.append(max_value)
    max_array = np.array(global_max)
    max_value = max_array.max()
    return max_value

# Function for loading and processing depthmaps.


def tf_load_pickle(path, max_value):
    def py_load_pickle(path, max_value):
        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = preprocess_depthmap(depthmap)
        depthmap = depthmap / max_value
        depthmap = tf.image.resize(depthmap, (image_target_height, image_target_width))
        targets = preprocess_targets(targets, targets_indices)
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
    depthmap.set_shape((image_target_height, image_target_width, 1))
    targets.set_shape((len(targets_indices,)))
    return depthmap, targets


# Parameters for dataset generation.
shuffle_buffer_size = 64
subsample_size = 1024
channels = list(range(0, 3))
targets_indices = [0]  # 0 is height, 1 is weight.

# Create dataset for training.
paths = paths_training
dataset_current = tf.data.Dataset.from_tensor_slices(paths)
dataset_current = dataset_current.map(lambda path: calculate_pickle(path))
dataset_max = get_max(dataset_current)
logging.info('Maximum value of training dataset: %.2f', dataset_max)
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_pickle(path, dataset_max))
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_training = dataset_norm
del dataset_norm

# Create dataset for validation.
# Note: No shuffle necessary.
paths = paths_validate
dataset_current = tf.data.Dataset.from_tensor_slices(paths)
dataset_current = dataset_current.map(lambda path: calculate_pickle(path))
dataset_max = get_max(dataset_current)
logging.info('Maximum value of validation dataset: %.2f', dataset_max)
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_pickle(path, dataset_max))
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validation = dataset_norm
del dataset_norm

#Create dataset for validation
paths = paths_activate
dataset_current = tf.data.Dataset.from_tensor_slices(paths)
dataset_current = dataset_current.map(lambda path: calculate_pickle(path))
dataset_max = get_max(dataset_current)
logging.info('Maximum value of activation dataset: %.2f', dataset_max)
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_pickle(path, dataset_max))
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_activation = dataset_norm
del dataset_norm

# Note: Now the datasets are prepared.

# Instantiate model.
model = models.Sequential()

model.add(
    layers.Conv2D(
        filters=32,
        kernel_size=(
            3,
            3),
        padding="same",
        activation="relu",
        input_shape=(
                image_target_height,
                image_target_width,
            1)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.02))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.04))

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.05))

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.06))

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.08))

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.09))

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dropout(0.1))

model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.2))

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


class GRADCamLogger(tf.keras.callbacks.Callback):
    def __init__(self, activation_data, layer_name, save_dir):
        super(GRADCamLogger, self).__init__()
        self.activation_data = activation_data
        self.layer_name = layer_name
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs):
        images = []
        grad_cam = []
      ## Initialize GRADCam Class
        cam = GradCAM(self.model, self.layer_name)
        count = 0
        foldername = self.save_dir + '/epoch{}'.format(epoch)
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        for data in self.activation_data:
            image = data[0]
            image = np.expand_dims(image, 0)
            pred = model.predict(image)
            classIDx = np.argmax(pred[0])

        ## Compute Heatmap
            heatmap = cam.compute_heatmap(image, classIDx)
            image = image.reshape(image.shape[1:])
            image = image * 255
            image = image.astype(np.uint8)

        ## Overlay heatmap on original image
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            implot = plt.imshow(np.squeeze(image))
            plt.imshow(heatmap, alpha=.6, cmap='inferno')
            plt.axis('off')
            plt.savefig(self.save_dir + '/epoch{}/out{}.png'.format(epoch, count),
                        bbox_inches='tight', transparent=True, pad_inches=0)
            plt.clf()
            count += 1
        make_grid(foldername)


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
#best_model_path = os.path.join('validation','best_model.h5')
best_model_path = './outputs/best_model.h5'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
training_callbacks.append(checkpoint_callback)

layer_name = 'conv2d_11'
#save_dir = os.path.join('validation','out')
save_dir = './outputs/out'
CHECK_FOLDER = os.path.isdir(save_dir)
#if not CHECK_FOLDER:
#    os.makedirs(save_dir)
#    logging.info('created folder : %s', save_dir)
#cam_callback = GRADCamLogger(dataset_activation,layer_name,save_dir)
#training_callbacks.append(cam_callback)

# Compile the model.
model.compile(
    optimizer="nadam",
    loss="mse",
    metrics=["mae"]
)

#es = tf.keras.callbacks.EarlyStopping(
#    monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='min')
#training_callbacks.append(es)

model.fit(
    dataset_training.batch(batch_size),
    validation_data=dataset_validation.batch(batch_size),
    epochs=epochs,
    callbacks=training_callbacks
)

run.upload_file(name=best_model_path, path_or_stream=best_model_path)

# Save the weights.
#logging.info('Saving and uploading weights...')
#path = "cnndepthmap_weights.h5"
#model.save_weights(path)
#run.upload_file(name="cnndepthmap_weights.h5", path_or_stream=path)

# Save the model.
#logging.info('Saving and uploading model...')
#path = "cnndepthmap_model"
#model.save(path)
#run.upload_folder(name="cnndepthmap", path=path)


# Done.
run.complete()
