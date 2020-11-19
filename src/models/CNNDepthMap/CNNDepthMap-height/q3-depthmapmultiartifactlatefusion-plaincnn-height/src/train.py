from pathlib import Path
import os
import random
import shutil

import glob2 as glob
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
from tensorflow.keras import callbacks, layers, models

from config import CONFIG, DATASET_MODE_DOWNLOAD, DATASET_MODE_MOUNT
from constants import DATA_DIR_ONLINE_RUN, MODEL_CKPT_FILENAME, REPO_DIR
from model import create_head, get_base_model
from augmentation import tf_augment_sample
from preprocessing_multiartifact import tf_load_pickle

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

from tmp_model_util.preprocessing import create_samples  # noqa: E402
from tmp_model_util.utils import download_dataset, get_dataset_path, AzureLogCallback, create_tensorboard_callback, get_optimizer  # noqa: E402

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
        dataset_path = run.input_datasets["dataset"]
    elif CONFIG.DATASET_MODE == DATASET_MODE_DOWNLOAD:
        dataset_path = get_dataset_path(DATA_DIR_ONLINE_RUN, dataset_name)
        download_dataset(workspace, dataset_name, dataset_path)
    else:
        raise NameError(f"Unknown DATASET_MODE: {CONFIG.DATASET_MODE}")

# Get the QR-code paths.
dataset_scans_path = os.path.join(dataset_path, "scans")
print("Dataset path:", dataset_scans_path)
# print(glob.glob(os.path.join(dataset_scans_path, "*"))) # Debug
print("Getting QR-code paths...")
qrcode_paths = glob.glob(os.path.join(dataset_scans_path, "*"))
print("qrcode_paths: ", len(qrcode_paths))
assert len(qrcode_paths) != 0

# Shuffle and split into train and validate.
random.seed(CONFIG.SPLIT_SEED)
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

paths_training = create_samples(qrcode_paths_training, CONFIG)
print(f"Samples for training: {len(paths_training)}")

paths_validate = create_samples(qrcode_paths_validate, CONFIG)
print(f"Samples for validate: {len(paths_validate)}")

# Create dataset for training.
paths = paths_training  # list
dataset = tf.data.Dataset.from_tensor_slices(paths)  # TensorSliceDataset  # List[ndarray[str]]
dataset = dataset.cache()
dataset = dataset.repeat(CONFIG.N_REPEAT_DATASET)
dataset = dataset.map(
    lambda path: tf_load_pickle(paths=path),
    tf.data.experimental.AUTOTUNE
)  # (240,180,5), (1,)

dataset = dataset.map(tf_augment_sample, tf.data.experimental.AUTOTUNE)

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(CONFIG.SHUFFLE_BUFFER_SIZE)
dataset_training = dataset

# Create dataset for validation.
# Note: No shuffle necessary.
paths = paths_validate
dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset_norm = dataset.map(lambda path: tf_load_pickle(path), tf.data.experimental.AUTOTUNE)
dataset_norm = dataset_norm.cache()
dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
dataset_validation = dataset_norm
del dataset_norm

# Note: Now the datasets are prepared.


# Create the base model
base_model = get_base_model(workspace, DATA_DIR)
base_model.summary()
assert base_model.output_shape == (None, 128)

# Create the head
head_input_shape = (128 * CONFIG.N_ARTIFACTS,)
head_model = create_head(head_input_shape, dropout=CONFIG.USE_CROPOUT)

# Implement artifact flow through the same model
model_input = layers.Input(
    shape=(CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, CONFIG.N_ARTIFACTS)
)

features_list = []
for i in range(CONFIG.N_ARTIFACTS):
    features_part = model_input[:, :, :, i:i + 1]
    features_part = base_model(features_part)
    features_list.append(features_part)

concatenation = tf.keras.layers.concatenate(features_list, axis=-1)
assert concatenation.shape.as_list() == tf.TensorShape((None, 128 * CONFIG.N_ARTIFACTS)).as_list()
model_output = head_model(concatenation)

model = models.Model(model_input, model_output)
model.summary()

best_model_path = str(DATA_DIR / f'outputs/{MODEL_CKPT_FILENAME}')
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
training_callbacks = [
    AzureLogCallback(run),
    create_tensorboard_callback(),
    checkpoint_callback,
]

optimizer = get_optimizer(CONFIG.USE_ONE_CYCLE,
                          lr=CONFIG.LEARNING_RATE,
                          n_steps=len(paths_training) / CONFIG.BATCH_SIZE)

# Compile the model.
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

# Train the model.
model.fit(
    dataset_training.batch(CONFIG.BATCH_SIZE),
    validation_data=dataset_validation.batch(CONFIG.BATCH_SIZE),
    epochs=CONFIG.EPOCHS,
    callbacks=training_callbacks,
    verbose=2
)

if CONFIG.EPOCHS_TUNE:
    # Un-freeze
    for layer in base_model._layers:
        layer.trainable = True

    # Adjust learning rate
    optimizer = tf.keras.optimizers.Nadam(learning_rate=CONFIG.LEARNING_RATE_TUNE)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    print("Start fine-tuning")
    model.fit(
        dataset_training.batch(CONFIG.BATCH_SIZE),
        validation_data=dataset_validation.batch(CONFIG.BATCH_SIZE),
        epochs=CONFIG.EPOCHS_TUNE,
        callbacks=training_callbacks,
        verbose=2
    )

# Done.
run.complete()
