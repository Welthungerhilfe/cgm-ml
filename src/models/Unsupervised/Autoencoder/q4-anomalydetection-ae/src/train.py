from pathlib import Path
import os
import pickle
import random
import shutil

import glob2 as glob
import tensorflow as tf
import tensorflow_datasets as tfds
from azureml.core import Experiment, Workspace
from azureml.core.run import Run

from config import CONFIG, CONFIG_DEV
from config import DATASET_MODE_DOWNLOAD, DATASET_MODE_MOUNT
from constants import DATA_DIR_ONLINE_RUN, MODEL_CKPT_FILENAME, REPO_DIR

from azureml.core.authentication import InteractiveLoginAuthentication


# Get the current run.
run = Run.get_context()

if run.id.startswith("OfflineRun"):
    CONFIG = CONFIG_DEV

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

print(f"Config: {CONFIG.NAME}")


from model import VariationalAutoencoder 
from tmp_model_util.preprocessing import preprocess_depthmap, preprocess_targets  # noqa: E402
from tmp_model_util.utils import download_dataset, get_dataset_path, AzureLogCallback, create_tensorboard_callback  # noqa: E402

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
    #interactive_auth = InteractiveLoginAuthentication(tenant_id="3a27c573-ec1a-4734-9cd3-3208af51794b")
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
        dataset_path = run.input_datasets["dataset"]
    elif CONFIG.DATASET_MODE == DATASET_MODE_DOWNLOAD:
        dataset_path = get_dataset_path(DATA_DIR_ONLINE_RUN, dataset_name)
        download_dataset(workspace, dataset_name, dataset_path)
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

# Restricting to a fixed number of scans.
print("Slicing scans...")
qrcode_paths = qrcode_paths[:CONFIG.DATASET_MAX_SCANS]
print("qrcode_paths: ", len(qrcode_paths))
assert len(qrcode_paths) == CONFIG.DATASET_MAX_SCANS

# Shuffle and split into train and validate.
random.shuffle(qrcode_paths)
split_index = int(len(qrcode_paths) * 0.8)
qrcode_paths_training = qrcode_paths[:split_index]
qrcode_paths_validate = qrcode_paths[split_index:]

del qrcode_paths

# Show split.
print("Paths for training:")
print(len(qrcode_paths_training))
print("\t" + "\n\t".join(qrcode_paths_training))
print("Paths for validation:")
print(len(qrcode_paths_validate))
print("\t" + "\n\t".join(qrcode_paths_validate))
assert len(qrcode_paths_training) > 0 and len(qrcode_paths_validate) > 0


# For mapping scan paths to samples.
# Files are shuffled per scan.
# And restricted to a maximum of samples per scan.
def get_files(scan_paths, extension="jpg"):
    result_paths = []
    for scan_path in scan_paths:
        sample_paths = glob.glob(os.path.join(scan_path, "**", "*." + extension))
        random.shuffle(sample_paths)
        sample_paths = sample_paths[:CONFIG.DATASET_MAX_SAMPLES_PER_SCAN]
        result_paths.extend(sample_paths)
    return result_paths


# Get the samples.
print("Getting paths...")
paths_training = get_files(qrcode_paths_training)
paths_validate = get_files(qrcode_paths_validate)
print("Using {} files for training.".format(len(paths_training)))
print("Using {} files for validation.".format(len(paths_validate)))
assert paths_training != 0
assert paths_validate != 0

# Shuffle all.
random.shuffle(paths_training)
random.shuffle(paths_validate)

# QR codes not necessary anymore.
del qrcode_paths_training
del qrcode_paths_validate

# Function for loading and processing images.
def tf_load_sample(path):
    def py_load_sample(path):
        image_string = tf.io.read_file(path)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image = tf.image.resize(image, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH))
        image = tf.image.rot90(image, k=3)
        image = image / 255.0
        return image

    image = tf.py_function(py_load_sample, [path], [tf.float32])[0]
    image.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 3))
    return image

# Create dataset for training.
# Note: Model will do caching et cetera.
paths = paths_training
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_sample(path))
dataset_train = dataset_norm
del dataset_norm

# Create dataset for validation.
# Note: No shuffle necessary.
# Note: Model will do caching et cetera.
paths = paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_sample(path))
dataset_validate = dataset_norm
del dataset_norm

# Create dataset for anomaly detection.
# Note: Model will do caching et cetera.
dataset_anomaly = tfds.load("cats_vs_dogs", split="train[:10%]")
def tf_preprocess(image):
    image = tf.image.resize(image, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH)) / 255.0
    return image
dataset_anomaly = dataset_anomaly.map(lambda sample: tf_preprocess(sample["image"]))

# Note: Now the datasets are prepared.

# Create the model.
model = VariationalAutoencoder(
    input_shape=(CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, CONFIG.IMAGE_TARGET_DEPTH), 
    filters=CONFIG.FILTERS, 
    latent_dim=CONFIG.LATENT_DIM,
    size=CONFIG.MODEL_SIZE
)
#model.summary()

# Make sure that output path exists.
outputs_path = "outputs"
if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)

# TODO Make some checkpoints work.
#best_model_path = str(DATA_DIR / f'outputs/{MODEL_CKPT_FILENAME}')
#checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#    filepath=best_model_path,
#    monitor="val_loss",
#    save_best_only=True,
#    verbose=1
#)
training_callbacks = [
    AzureLogCallback(run)
#    create_tensorboard_callback(),
#    checkpoint_callback,
]

# Train the model.
model.train(
    dataset_train, 
    dataset_validate,
    dataset_anomaly, 
    epochs=CONFIG.EPOCHS,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle_buffer_size=CONFIG.SHUFFLE_BUFFER_SIZE,
    render=CONFIG.RENDER,
    render_every=5,
    callbacks=training_callbacks,
    outputs_path=outputs_path
    #kl_loss_factor=CONFIG.KL_LOSS_FACTOR
)

# Done.
run.complete()
