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


def get_prediction_uncertainty(model_path: str, dataset_evaluation: tf.data.Dataset, dropout_strength: float, num_dropout_predictions: int) -> np.array:
    """Predict standard deviation of multiple predictions with different dropouts

    Args:
        model_path: Path of the trained model
        dataset_evaluation: dataset in which the evaluation need to performed

    Returns:
        predictions, array shape (N_SAMPLES, )
    """
    logging.info("loading model from %s", model_path)
    model = load_model(model_path, compile=False)
    model = change_dropout_strength(model, dropout_strength)

    dataset = dataset_evaluation.batch(1)

    logging.info("starting predicting uncertainty")
    start = time.time()
    std_list = [predict_uncertainty(X, model, num_dropout_predictions) for X, y in dataset.as_numpy_iterator()]
    end = time.time()
    logging.info("Total time for uncertainty prediction experiment: %.2f sec", end - start)

    return np.array(std_list)


def predict_uncertainty(X: np.array, model: tf.keras.Model, num_dropout_predictions: int) -> float:
    """Predict standard deviation of multiple predictions with different dropouts
    Args:
        X: Sample image with shape (1, h, w, 1)
        model: keras model

    Returns:
        The standard deviation of multiple predictions
    """
    one_batch = np.repeat(X, num_dropout_predictions, axis=0)
    predictions = model(one_batch, training=True)
    std = tf.math.reduce_std(predictions)
    return std


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
