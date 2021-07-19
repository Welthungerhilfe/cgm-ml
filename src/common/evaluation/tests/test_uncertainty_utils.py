from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).parents[2]))  # common/ dir

from evaluation.uncertainty_utils import _predict, _calculate_std  # noqa: E402
from model_utils.model_plaincnn import create_cnn  # noqa: E402

BATCH_SIZE = 8


def test_predict_uncertainty():
    input_shape = (240, 180, 1)
    num_batches = 2

    model = create_cnn(input_shape, dropout=False)

    data_shape = (num_batches * BATCH_SIZE,) + input_shape
    train_examples = np.zeros(data_shape)
    train_labels = np.zeros(num_batches * BATCH_SIZE)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    dataset = train_dataset.batch(BATCH_SIZE)

    uncertainties = _predict(model, dataset)
    assert uncertainties.shape == (2 * BATCH_SIZE, 1), uncertainties.shape


def test_calculate_std_same_prediction():
    num_samples = 16
    predictions_per_model = [
        np.zeros(num_samples),
        np.zeros(num_samples),
    ]
    std = _calculate_std(predictions_per_model)
    assert np.all(std == np.zeros(num_samples))


def test_calculate_std_different_prediction():
    num_samples = 16
    predictions_per_model = [
        np.zeros(num_samples),
        np.ones(num_samples),
    ]
    std = _calculate_std(predictions_per_model)
    assert np.all(std == np.ones(num_samples) / 2)
