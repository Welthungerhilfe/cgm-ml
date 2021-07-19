import logging
import logging.config
import time
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def get_prediction_uncertainty_deepensemble(model_paths: list,
                                            dataset_evaluation: tf.data.Dataset,
                                            batch_size: int = 256) -> np.array:
    """Predict standard deviation of multiple predictions with different dropouts
    Args:
        model_path: Path of the trained model
        dataset_evaluation: dataset in which the evaluation need to performed
    Returns:
        predictions, array shape (N_SAMPLES, )
    """
    dataset = dataset_evaluation.batch(batch_size)
    logging.info("Start predicting uncertainty")

    # Go through all models and compute standard deviation of predictions
    start = time.time()
    predictions_list = [_load_and_predict(model_path, dataset) for model_path in model_paths]
    end = time.time()
    logging.info("Total time for uncertainty prediction experiment: %f:.3 sec", end - start)

    std = _calculate_std(predictions_list)
    return std


def _load_and_predict(model_path, dataset):
    logging.info("Loading model from %s", model_path)
    model = load_model(model_path, compile=False)
    logging.info("Predicting with model %s", model_path)
    return _predict(model, dataset)


def _predict(model: tf.Module, dataset: tf.data.Dataset) -> np.array:
    predictions_batches = []
    for X, _y in dataset.as_numpy_iterator():
        prediction_batch = model.predict(X)  # shape (BATCH_SIZE, 1)
        predictions_batches.append(prediction_batch)
    predictions = np.concatenate(predictions_batches)
    return predictions


def _calculate_std(predictions_per_model: List[np.array]) -> np.array:
    predictions_per_model_ = np.array(predictions_per_model)
    std = np.std(predictions_per_model_, axis=0)
    return std
