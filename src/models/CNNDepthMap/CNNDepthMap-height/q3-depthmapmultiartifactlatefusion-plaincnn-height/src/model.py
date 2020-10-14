import os
from pathlib import Path

from azureml.core import Experiment, Workspace
from azureml.core.run import Run
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from config import CONFIG
from constants import REPO_DIR

run = Run.get_context()
DATA_DIR = REPO_DIR / 'data' if run.id.startswith("OfflineRun") else Path(".")


def get_head_model():
    head_input_shape = (128 * CONFIG.N_ARTIFACTS,)
    return create_head(head_input_shape, dropout=CONFIG.USE_CROPOUT)


def get_base_model():
    if CONFIG.PRETRAINED_RUN:
        model_fpath = DATA_DIR / "pretrained/" / CONFIG.PRETRAINED_RUN / "best_model.h5"
        if not os.path.exists(model_fpath):
            download_pretrained_model(model_fpath)
        print(f"Loading pretrained model from {model_fpath}")
        base_model = load_base_cgm_model(model_fpath, should_freeze=CONFIG.SHOULD_FREEZE_BASE)
    else:
        input_shape = (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1)
        base_model = create_base_cnn(input_shape, dropout=CONFIG.USE_DROPOUT)  # output_shape: (128,)
    return base_model


def download_pretrained_model(output_model_fpath):
    print(f"Downloading pretrained model from {CONFIG.PRETRAINED_RUN}")
    previous_experiment = Experiment(workspace=Workspace.from_config(), name=CONFIG.PRETRAINED_EXPERIMENT)
    previous_run = Run(previous_experiment, CONFIG.PRETRAINED_RUN)
    previous_run.download_file("outputs/best_model.h5", output_model_fpath)


# adapted from https://github.com/AI-Guru/ngdlm/blob/master/ngdlm/models/gan.py
class LateFusionModel(Model):
    def __init__(self, base_model, head_model):
        super().__init__()

        assert base_model != None
        assert head_model != None

        self.base_model = base_model
        self.head_model = head_model

        # Implement artifact flow through the same model
        model_input = layers.Input(
            shape=(CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, CONFIG.N_ARTIFACTS)
        )

        features_list = []
        for i in range(CONFIG.N_ARTIFACTS):
            features_part = model_input[:, :, :, i:i + 1]
            features_part = base_model(features_part)
            features_list.append(features_part)

        concatenation = tf.concat(features_list, axis=-1)
        model_output = layers.Dense(1, activation="linear")(concatenation)  # shape: (None,640)

        self.latefusion = models.Model(model_input, model_output)

    def compile(self, optimizer, loss="mse", metrics=["mae"], **kwargs):
        """Compiles the model. Same as vanilla Keras"""
        self.latefusion.compile(optimizer, loss, metrics, **kwargs)

    def fit(self, x, **kwargs):
        """Compiles the model. Same as vanilla Keras"""
        self.latefusion.fit(x, **kwargs)

    def summary(self):
        print("base_model:")
        self.base_model.summary()
        print("head_model:")
        self.head_model.summary()
        print("latefusion:")
        self.latefusion.summary()

    def save(self, filepath):
        """Save latefusion model including the base net and the head.

        The base and head use the path plus a respective annotation.
        This code

        >>> latefusionmodel = LateFusionModel(base_model, head_model)
        >>> latefusionmodel.save("latefusionmodel.h5")

        will create the files *latefusionmodel.h5*,
                              *latefusionmodel-base.h5*, and
                              *latefusionmodel-head.h5*.
        """
        print("save()")
        self.latefusion.save(filepath)
        self.base_model.save(append_to_filepath(filepath, "-base"))
        self.head_model.save(append_to_filepath(filepath, "-head"))

    def save_weights(self, filepath):
        print("save_weights()")
        self.latefusion.save_weights(filepath)
        self.base_model.save_weights(append_to_filepath(filepath, "-base-weights"))
        self.head_model.save_weights(append_to_filepath(filepath, "-head-weights"))

    def load_weights(self, filepath, **kwargs):
        self.latefusion.load_weights(filepath, **kwargs)
        self.base_model.load_weights(append_to_filepath(filepath, "-base"), **kwargs)
        self.head_model.load_weights(append_to_filepath(filepath, "-head"), **kwargs)


def load_base_cgm_model(model_fpath, should_freeze=False):
    # load model
    loaded_model = models.load_model(model_fpath)

    # cut off last layer
    _ = loaded_model._layers.pop()

    if should_freeze:
        for layer in loaded_model._layers:
            layer.trainable = False

    return loaded_model


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


def append_to_filepath(filepath, string):
    """Add a string to a file-path. Right before the extension."""
    filepath, extension = os.path.splitext(filepath)
    return filepath + string + extension
