import copy
import logging
import logging.config
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.python import keras


def change_dropout_strength(model: tf.keras.Model, dropout_strength: float) -> tf.keras.Model:
    """Duplicate a model while adjusting the dropout rate"""
    new_model = Sequential(name="new_model")
    for layer_ in model.layers:
        layer = copy.copy(layer_)
        if isinstance(layer, keras.layers.core.Dropout):
            # Set the dropout rate a ratio from range [0.0, 1.0]
            layer.rate = min(0.999, layer.rate * dropout_strength)
        new_model.add(layer)
    return new_model


def get_prediction_uncertainty_deepensemble(model_paths: list, dataset_evaluation: tf.data.Dataset) -> np.array:
    """Predict standard deviation of multiple predictions with different dropouts
    Args:
        model_path: Path of the trained model
        dataset_evaluation: dataset in which the evaluation need to performed
    Returns:
        predictions, array shape (N_SAMPLES, )
    """

    dataset = dataset_evaluation.batch(1)

    logging.info("starting predicting uncertainty")

    # Go through all models and compute STD of predictions.
    start = time.time()
    std_list = []
    for model_path in model_paths:
        logging.info(f"loading model from {model_path}")
        model = load_model(model_path, compile=False)
        std_list += [[model.predict(X)[0] for X, y in dataset.as_numpy_iterator()]]
    std_list = np.array(std_list)
    std_list = np.std(std_list, axis=0)
    std_list = std_list.reshape((-1))
    end = time.time()
    logging.info(f"Total time for uncertainty prediction experiment: {end - start:.3} sec")

    return np.array(std_list)
