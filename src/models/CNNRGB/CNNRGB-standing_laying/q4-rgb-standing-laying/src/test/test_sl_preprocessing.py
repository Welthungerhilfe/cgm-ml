import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).parents[1]))  # noqa: E402

from sl_constants import REPO_DIR  # noqa: E402
from sl_preprocessing import process_path  # noqa: E402


def test_get_label_0():
    paths = [str(REPO_DIR) + '/src/models/CNNRGB/CNNRGB-standing_laying/q4-rgb-standing-laying/src/test/anon-rgb-classification/test/laying/rgb_1597886481-znzw5yzhjh_1597886481929_200_13520.886623759001.jpg']
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: process_path(path))

    for _, label in dataset.as_numpy_iterator():
        assert label == 0


def test_get_label_1():
    paths = [str(REPO_DIR) + '/src/models/CNNRGB/CNNRGB-standing_laying/q4-rgb-standing-laying/src/test/anon-rgb-classification/test/standing/rgb_1585352016-s51bhrzmtt_1592712865086_100_520010.772824759.jpg']
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: process_path(path))

    for _, label in dataset.as_numpy_iterator():
        assert label == 1


def test_data():
    paths = [str(REPO_DIR) + '/src/models/CNNRGB/CNNRGB-standing_laying/q4-rgb-standing-laying/src/test/anon-rgb-classification/test/laying/rgb_1597886481-znzw5yzhjh_1597886481929_200_13520.886623759001.jpg']
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: process_path(path))

    for a, _ in dataset.as_numpy_iterator():
        assert np.max(a) <= 1
        assert np.min(a) >= 0
