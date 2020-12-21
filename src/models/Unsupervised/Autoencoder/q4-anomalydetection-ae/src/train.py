from pathlib import Path
import os
import random
import shutil

import glob2 as glob
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run

from config import CONFIG
#from config import CONFIG_DEV as CONFIG #  Only for development.
from constants import REPO_DIR

# Get the current run.
run = Run.get_context()
offline_run = run.id.startswith("OfflineRun")

if offline_run:
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

from model import Autoencoder   # noqa: E402
from dataset import create_datasets   # noqa: E402
from tmp_model_util.utils import AzureLogCallback   # noqa: E402

# Make experiment reproducible
tf.random.set_seed(CONFIG.SPLIT_SEED)
random.seed(CONFIG.SPLIT_SEED)

if offline_run:
    print("Running in offline mode...")

    # Access workspace.
    print("Accessing workspace...")
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "training-junkyard")
    run = experiment.start_logging(outputs=None, snapshot_directory=None)

# Online run. Use dataset provided by training notebook.
else:
    print("Running in online mode...")
    experiment = run.experiment
    workspace = experiment.workspace

# Prepare the datasets.
dataset_train, dataset_validate, dataset_anomaly = create_datasets(workspace, experiment, run, offline_run, CONFIG)

# Create the model.
model = Autoencoder(
    family=CONFIG.MODEL_FAMILY,
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
