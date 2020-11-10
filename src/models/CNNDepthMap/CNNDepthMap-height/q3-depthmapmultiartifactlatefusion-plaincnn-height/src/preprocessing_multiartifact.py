import pickle
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from config import CONFIG


@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])   # List of length n_artifacts
def tf_load_pickle(paths):
    """Load and process depthmaps"""
    depthmap, targets = tf.py_function(create_multiartifact_sample, [paths], [tf.float32, tf.float32])
    depthmap.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, CONFIG.N_ARTIFACTS))
    targets.set_shape((len(CONFIG.TARGET_INDEXES,)))
    return depthmap, targets  # (240,180,5), (1,)


def create_multiartifact_sample(artifacts: List[str]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Open pickle files and load data.

    Args:
        artifacts: List of file paths to pickle files

    Returns:
        depthmaps of shape (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH, n_artifacts)
        targets of shape (1, )
    """
    targets_list = []
    n_artifacts = len(artifacts)
    depthmaps = np.zeros((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, n_artifacts))

    for i, artifact_path in enumerate(artifacts):
        depthmap, targets = py_load_pickle(artifact_path, CONFIG.NORMALIZATION_VALUE)
        depthmap.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1))
        depthmaps[:, :, i] = tf.squeeze(depthmap, axis=2)
        targets_list.append(targets)
    targets = targets_list[0]
    if not np.all(targets_list == targets):
        print("Warning: Not all targets are the same!!\n"
              f"target_list: {str(targets_list)} artifacts: {str(artifacts)}")

    return depthmaps, targets


def py_load_pickle(path, max_value):
    path_ = path if isinstance(path, str) else path.numpy()
    try:
        depthmap, targets = pickle.load(open(path_, "rb"))
    except OSError as e:
        print(f"path: {path}, type(path) {str(type(path))}")
        print(e)
        raise e
    depthmap = preprocess_depthmap(depthmap)
    depthmap = depthmap / max_value
    depthmap = tf.image.resize(depthmap, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH))
    targets = preprocess_targets(targets, CONFIG.TARGET_INDEXES)
    return depthmap, targets


def preprocess_depthmap(depthmap):
    # TODO here be more code.
    return depthmap.astype("float32")


def preprocess_targets(targets, targets_indices):
    if targets_indices is not None:
        targets = targets[targets_indices]
    return targets.astype("float32")
