import os
import logging
import logging.config
from pathlib import Path
import subprocess

import tensorflow as tf
import tensorflow_addons as tfa
from azureml.core.run import Run
from azureml.core.workspace import Workspace
from tensorflow.keras import callbacks

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


def get_optimizer(use_one_cycle: bool, lr: float,
                  n_steps: int) -> tf.python.keras.optimizer_v2.optimizer_v2.OptimizerV2:
    if use_one_cycle:
        lr_schedule = tfa.optimizers.TriangularCyclicalLearningRate(
            initial_learning_rate=lr / 100,
            maximal_learning_rate=lr,
            step_size=n_steps,
        )
        # Note: When using 1cycle, this uses the Adam (not Nadam) optimizer
        return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return tf.keras.optimizers.Nadam(learning_rate=lr)


def download_dataset(workspace: Workspace, dataset_name: str, dataset_path: str):
    logging.info("Accessing dataset...")
    if os.path.exists(dataset_path):
        return
    dataset = workspace.datasets[dataset_name]
    logging.info("Downloading dataset %s", dataset_name)
    dataset.download(target_path=dataset_path, overwrite=False)
    logging.info("Finished downloading %s", dataset_name)


def get_dataset_path(data_dir: Path, dataset_name: str) -> str:
    return str(data_dir / dataset_name)


class AzureLogCallback(callbacks.Callback):
    """Pushes metrics and losses into the run on AzureML"""
    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                self.run.log(key, value)


def create_tensorboard_callback() -> callbacks.TensorBoard:
    return callbacks.TensorBoard(
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


WANDB_API_KEY_MH = "237ca046c5dcd915945761dc477207549ef2c42c"


def setup_wandb():
    wandb_login = subprocess.run(["wandb", "login", WANDB_API_KEY_MH])
    assert wandb_login.returncode == 0
