import os
import random

import glob2 as glob
import numpy as np
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
from tensorflow.keras import callbacks

from config import CONFIG
from constants import REPO_DIR
from model import create_cnn, set_trainable_below_layers
from preprocessing import process_path

# Make experiment reproducable
tf.random.set_seed(CONFIG.SPLIT_SEED)
random.seed(CONFIG.SPLIT_SEED)

# Get the current run.
run = Run.get_context()

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if(run.id.startswith("OfflineRun")):
    print("Running in offline mode...")

    # Access workspace.
    print("Accessing workspace...")
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "training-junkyard")
    run = experiment.start_logging(outputs=None, snapshot_directory=None)

    # Get dataset.
    print("Accessing dataset...")
    dataset_name = "anon-rgb-classification"
    dataset_path = str(REPO_DIR / "data" / dataset_name)
    if not os.path.exists(dataset_path):
        dataset = workspace.datasets[dataset_name]
        dataset.download(target_path=dataset_path, overwrite=False)

# Online run. Use dataset provided by training notebook.
else:
    print("Running in online mode...")
    experiment = run.experiment
    workspace = experiment.workspace
    dataset_path = run.input_datasets["dataset"]

# Get the Image paths.
dataset_path = os.path.join(dataset_path, "test")
print("Dataset path:", dataset_path)
print("Getting image...")
image_paths = glob.glob(os.path.join(dataset_path, "*/*.jpg"))
print(len(image_paths))
print(tf.__version__)

assert len(image_paths) != 0

# Shuffle and split into train and validate.
random.shuffle(image_paths)
split_index = int(len(image_paths) * 0.8)
image_paths_training = image_paths[:split_index]
image_paths_validate = image_paths[split_index:]
image_paths_activation = random.choice(image_paths_validate)
image_paths_activation = [image_paths_activation]

del image_paths

# Show split.
print("Paths for training:")
print("\t" + "\n\t".join(image_paths_training))
print("Paths for validation:")
print("\t" + "\n\t".join(image_paths_validate))
print("Paths for activation:")
print("\t" + "\n\t".join(image_paths_activation))

print(len(image_paths_training))
print(len(image_paths_validate))

assert len(image_paths_training) > 0 and len(image_paths_validate) > 0

# Parameters for dataset generation.
class_names = np.array(sorted([item.split('/')[-1] for item in glob.glob(os.path.join(dataset_path, "*"))]))
print(class_names)

# Create dataset for training.
paths = image_paths_training
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: process_path(path))
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_norm = dataset_norm.shuffle(CONFIG.SHUFFLE_BUFFER_SIZE)
dataset_training = dataset_norm
del dataset_norm

# Create dataset for validation.
# Note: No shuffle necessary.
paths = image_paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: process_path(path))
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validation = dataset_norm
del dataset_norm

input_shape = (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 3)
model = create_cnn(input_shape, dropout=True)
model.summary()
print(len(model.trainable_weights))
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
#best_model_path = os.path.join('validation','best_model.h5')
best_model_path = './outputs/best_model.h5'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
training_callbacks.append(checkpoint_callback)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=CONFIG.LEARNING_RATE)

# Compile the model.
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
# Train the model.
model.fit(
    dataset_training.batch(CONFIG.BATCH_SIZE),
    validation_data=dataset_validation.batch(CONFIG.BATCH_SIZE),
    epochs=CONFIG.EPOCHS,
    callbacks=training_callbacks,
    verbose=2
)

#  function use to tune the top convolution layer
set_trainable_below_layers('block14_sepconv1', model)

model.fit(
    dataset_training.batch(CONFIG.BATCH_SIZE),
    validation_data=dataset_validation.batch(CONFIG.BATCH_SIZE),
    epochs=CONFIG.TUNE_EPOCHS,
    callbacks=training_callbacks,
    verbose=2
)


# Done.
run.complete()
