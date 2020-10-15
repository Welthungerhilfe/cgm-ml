from train import process_path
import numpy as np
from pathlib import Path
import sys
import tensorflow as tf

sys.path.append(str(Path(__file__).parents[1]))


def test_get_label_0():
    paths = ['/Users/prajwalsingh/cgm-ml-service/data/anon-rgb-classification/test/laying/rgb_1597851643-23im8oux3j_1597851643547_202_330741.97858975903.jpg']
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: process_path(path))

    for _, label in dataset.as_numpy_iterator():
        assert label == 0


def test_get_label_1():
    paths = ['/Users/prajwalsingh/cgm-ml-service/data/anon-rgb-classification/test/standing/rgb_1583438117-71v1y4z0gd_1592711198959_100_74914.611084559.jpg']
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: process_path(path))

    for _, label in dataset.as_numpy_iterator():
        assert label == 1


def test_data():
    paths = ['/Users/prajwalsingh/cgm-ml-service/data/anon-rgb-classification/test/laying/rgb_1597851643-23im8oux3j_1597851643547_202_330741.97858975903.jpg']
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: process_path(path))

    for a, _ in dataset.as_numpy_iterator():
        assert np.max(a) <= 1
        assert np.min(a) >= 0 and np.min(a) >= 0
