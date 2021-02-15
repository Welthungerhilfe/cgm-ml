from pathlib import Path
import os
import pickle
import random
import shutil
import sys

import glob2 as glob
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
import wandb
from wandb.keras import WandbCallback
import evidential_deep_learning as edl

from config import CONFIG, DATASET_MODE_DOWNLOAD, DATASET_MODE_MOUNT
from constants import DATA_DIR_ONLINE_RUN, MODEL_CKPT_FILENAME, REPO_DIR

# Get the current run.
run = Run.get_context()

if run.id.startswith("OfflineRun"):
    utils_dir_path = REPO_DIR / "src/common/model_utils"
    utils_paths = glob.glob(os.path.join(utils_dir_path, "*.py"))
    temp_model_util_dir = Path(__file__).parent / "tmp_model_util"
    # Remove old temp_path
    if os.path.exists(temp_model_util_dir):
        shutil.rmtree(temp_model_util_dir)
    # Copy
    os.mkdir(temp_model_util_dir)
    os.system(f'touch {temp_model_util_dir}/__init__.py')
    for p in utils_paths:
        shutil.copy(p, temp_model_util_dir)

from model import create_cnn  # noqa: E402
from tmp_model_util.preprocessing import preprocess_depthmap, preprocess_targets  # noqa: E402
from tmp_model_util.utils import calculate_mean_and_std_targets, download_dataset, get_dataset_path, AzureLogCallback, create_tensorboard_callback, get_optimizer, setup_wandb  # noqa: E402

# Make experiment reproducible
tf.random.set_seed(CONFIG.SPLIT_SEED)
random.seed(CONFIG.SPLIT_SEED)

DATA_DIR = REPO_DIR / 'data' if run.id.startswith("OfflineRun") else Path(".")
print(f"DATA_DIR: {DATA_DIR}")

# Offline run. Download the sample dataset and run locally. Still push results to Azure.
if run.id.startswith("OfflineRun"):
    print("Running in offline mode...")

    # Access workspace.
    print("Accessing workspace...")
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "training-junkyard")
    run = experiment.start_logging(outputs=None, snapshot_directory=None)

    dataset_name = CONFIG.DATASET_NAME_LOCAL
    dataset_path = get_dataset_path(DATA_DIR, dataset_name)
    download_dataset(workspace, dataset_name, dataset_path)

# Online run. Use dataset provided by training notebook.
else:
    print("Running in online mode...")
    experiment = run.experiment
    workspace = experiment.workspace

    dataset_name = CONFIG.DATASET_NAME

    # Mount or download
    if CONFIG.DATASET_MODE == DATASET_MODE_MOUNT:
        dataset_path = sys.argv[1]  # This expects the dataset_path to be the first argument to this script
    elif CONFIG.DATASET_MODE == DATASET_MODE_DOWNLOAD:
        print('Run', run)
        print("input_datasets", run.input_datasets)
        print("sys.argv[1]", sys.argv[1])
        dataset_path = sys.argv[1]
        # dataset_path = run.input_datasets['input_1']

        # dataset_path = get_dataset_path(DATA_DIR_ONLINE_RUN, dataset_name)
        # download_dataset(workspace, dataset_name, dataset_path)
    else:
        raise NameError(f"Unknown DATASET_MODE: {CONFIG.DATASET_MODE}")

# Get the QR-code paths.
dataset_path = os.path.join(dataset_path, "scans")
print("Dataset path:", dataset_path)
#print(glob.glob(os.path.join(dataset_path, "*"))) # Debug
print("Getting QR-code paths...")
qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
print("qrcode_paths: ", len(qrcode_paths))
assert len(qrcode_paths) != 0

# Shuffle and split into train and validate.
random.shuffle(qrcode_paths)
split_index = int(len(qrcode_paths) * 0.8)
qrcode_paths_training = qrcode_paths[:split_index]
qrcode_paths_validate = qrcode_paths[split_index:]

del qrcode_paths

# Show split.
print("Paths for training:")
print("\t" + "\n\t".join(qrcode_paths_training))
print("Paths for validation:")
print("\t" + "\n\t".join(qrcode_paths_validate))

print(len(qrcode_paths_training))
print(len(qrcode_paths_validate))

assert len(qrcode_paths_training) > 0 and len(qrcode_paths_validate) > 0


def get_depthmap_files(paths):
    pickle_paths = []
    for path in paths:
        pickle_paths.extend(glob.glob(os.path.join(path, "**", "*.p")))
    return pickle_paths


# Get the pointclouds.
print("Getting depthmap paths...")
paths_training = get_depthmap_files(qrcode_paths_training)
paths_validate = get_depthmap_files(qrcode_paths_validate)

del qrcode_paths_training
del qrcode_paths_validate

print("Using {} files for training.".format(len(paths_training)))
print("Using {} files for validation.".format(len(paths_validate)))


# Function for loading and processing depthmaps.
def tf_load_pickle(path, max_value):
    def py_load_pickle(path, max_value):
        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = preprocess_depthmap(depthmap)
        depthmap = depthmap / max_value
        mean = 0.18
        std = 0.07
        depthmap = (depthmap - mean) / std
        depthmap = tf.image.resize(depthmap, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH))
        targets = preprocess_targets(targets, CONFIG.TARGET_INDEXES)
        mean = 91.0
        std = 9.7
        targets = (targets - mean) / std
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
    depthmap.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1))
    targets.set_shape((len(CONFIG.TARGET_INDEXES,)))
    return depthmap, targets


def tf_flip(image):
    image = tf.image.random_flip_left_right(image)
    return image


# Create dataset for training.
paths = paths_training
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_norm = dataset_norm.shuffle(CONFIG.SHUFFLE_BUFFER_SIZE)
dataset_training = dataset_norm
del dataset_norm

# mean, std, minimum = calculate_mean_and_std_targets(dataset_training, 20000)

# Create dataset for validation.
# Note: No shuffle necessary.
paths = paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_pickle(path, CONFIG.NORMALIZATION_VALUE))
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validation = dataset_norm
del dataset_norm

# Note: Now the datasets are prepared.

# Create the model.
input_shape = (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1)
model = create_cnn(input_shape, dropout=CONFIG.USE_DROPOUT)
model.summary()

best_model_path = str(DATA_DIR / f'outputs/{MODEL_CKPT_FILENAME}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

dataset_batches = dataset_training.batch(CONFIG.BATCH_SIZE)

training_callbacks = [
    AzureLogCallback(run),
    create_tensorboard_callback(),
    checkpoint_callback,
]

if getattr(CONFIG, 'USE_WANDB', False):
    setup_wandb()
    wandb.init(project="ml-project", entity="cgm-team")
    wandb.config.update(CONFIG)
    training_callbacks.append(WandbCallback(log_weights=True, log_gradients=True, training_data=dataset_batches))

optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.LEARNING_RATE)
# optimizer = get_optimizer(CONFIG.USE_ONE_CYCLE,
#                           lr=CONFIG.LEARNING_RATE,
#                           n_steps=len(paths_training) / CONFIG.BATCH_SIZE)

def EvidentialRegressionLoss(true, pred):
    return edl.losses.EvidentialRegression(true, pred, coeff=CONFIG.EDL_COEFF)


# Compile the model.
model.compile(
    optimizer=optimizer,
    loss=EvidentialRegressionLoss,
    metrics=["mae"]
)

# Train the model.
model.fit(
    dataset_training.batch(CONFIG.BATCH_SIZE),
    validation_data=dataset_batches,
    epochs=CONFIG.EPOCHS,
    callbacks=training_callbacks,
    verbose=2
)

# Done.
run.complete()
