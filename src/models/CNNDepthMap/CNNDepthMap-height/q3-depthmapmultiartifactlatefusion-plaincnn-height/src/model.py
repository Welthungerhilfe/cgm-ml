import os
from pathlib import Path

from azureml.core import Experiment, Workspace
from azureml.core.run import Run
from tensorflow.keras import models, layers

from config import CONFIG
from constants import MODEL_H5_FILENAME


def get_base_model(workspace: Workspace, data_dir: Path) -> models.Sequential:
    if CONFIG.PRETRAINED_RUN:
        model_fpath = data_dir / "pretrained/" / CONFIG.PRETRAINED_RUN / MODEL_H5_FILENAME
        if not os.path.exists(model_fpath):
            download_pretrained_model(workspace, model_fpath)
        print(f"Loading pretrained model from {model_fpath}")
        base_model = load_base_cgm_model(model_fpath, should_freeze=CONFIG.SHOULD_FREEZE_BASE)
    else:
        input_shape = (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1)
        base_model = create_base_cnn(input_shape, dropout=CONFIG.USE_DROPOUT)  # output_shape: (128,)
    return base_model


def download_pretrained_model(workspace: Workspace, output_model_fpath: str):
    print(f"Downloading pretrained model from {CONFIG.PRETRAINED_RUN}")
    previous_experiment = Experiment(workspace=workspace, name=CONFIG.PRETRAINED_EXPERIMENT)
    previous_run = Run(previous_experiment, CONFIG.PRETRAINED_RUN)
    previous_run.download_file(f"outputs/{MODEL_H5_FILENAME}", output_model_fpath)


def load_base_cgm_model(model_fpath: str, should_freeze: bool=False) -> models.Sequential:
    # load model
    loaded_model = models.load_model(model_fpath)

    # cut off last layer (https://stackoverflow.com/a/59304656/5497962)
    beheaded_model = models.Sequential(name="base_model_beheaded")
    for layer in loaded_model.layers[:-1]:
        beheaded_model.add(layer)

    if should_freeze:
        for layer in beheaded_model._layers:
            layer.trainable = False

    return beheaded_model


def create_base_cnn(input_shape, dropout):
    model = models.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(layers.Dropout(0.05))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(layers.Dropout(0.075))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(layers.Dropout(0.125))

    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(layers.Dropout(0.15))

    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(layers.Dropout(0.175))

    model.add(layers.Flatten())

    model.add(layers.Dense(1024, activation="relu"))
    if dropout:
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(128, activation="relu"))
    if dropout:
        model.add(layers.Dropout(0.25))

    return model


def create_head(input_shape, dropout):
    model = models.Sequential(name="head")
    model.add(layers.Dense(128, activation="relu", input_shape=input_shape))
    if dropout:
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64, activation="relu"))
    if dropout:
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(16, activation="relu"))
    if dropout:
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1, activation="linear"))
    return model
