import datetime
import os
from pathlib import Path

from azureml.core.run import Run
from azureml.core.workspace import Workspace
from tensorflow.keras import callbacks


def download_dataset(workspace: Workspace, dataset_name: str, dataset_path: str):
    print("Accessing dataset...")
    if os.path.exists(dataset_path):
        return
    dataset = workspace.datasets[dataset_name]
    print("Downloading dataset.. Current date and time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    dataset.download(target_path=dataset_path, overwrite=False)
    print("Finished downloading, Current date and time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def get_dataset_path(data_dir: Path, dataset_name: str):
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
