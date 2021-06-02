import logging
import logging.config
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def get_prediction_uncertainty_deepensemble(model_paths: list, dataset_evaluation: tf.data.Dataset) -> np.array:
    """Predict standard deviation of multiple predictions with different dropouts
    Args:
        model_path: Path of the trained model
        dataset_evaluation: dataset in which the evaluation need to performed
    Returns:
        predictions, array shape (N_SAMPLES, )
    """
    dataset = dataset_evaluation.batch(1)
    logging.info("Start predicting uncertainty")

    # Go through all models and compute STD of predictions.
    start = time.time()
    std_list = []
    for model_path in model_paths:
        logging.info("Loading model from %s", model_path)
        model = load_model(model_path, compile=False)
        logging.info("Predicting with model %s", model_path)
        std_list += [[model.predict(X)[0] for X, _y in dataset.as_numpy_iterator()]]
    std_list = np.array(std_list)
    std_list = np.std(std_list, axis=0)
    std_list = std_list.reshape((-1))
    end = time.time()
    logging.info("Total time for uncertainty prediction experiment: %f:.3 sec", end - start)

    return np.array(std_list)
