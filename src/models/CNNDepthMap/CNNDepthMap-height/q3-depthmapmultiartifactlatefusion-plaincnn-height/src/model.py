from tensorflow.keras import models, layers


from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from config import CONFIG
from constants import REPO_DIR

DATA_DIR = REPO_DIR / 'data' if run.id.startswith("OfflineRun") else Path(".")

def main():
    base_model = get_base_model()

    head_input_shape = (128 * CONFIG.N_ARTIFACTS,)
    head_model = create_head(head_input_shape, dropout=CONFIG.USE_CROPOUT)

    LateFusionModel(base_model, head_model)


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
    model.add(layers.Dense(128, activation="relu"))
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
