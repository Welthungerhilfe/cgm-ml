import glob2 as glob
from functools import partial
from itertools import groupby, islice
import os
import pickle
import re
from typing import Iterator, List, Tuple

import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf

from config import (CONFIG, DATA_AUGMENTATION_SAME_PER_CHANNEL,
                    DATA_AUGMENTATION_NO, DATA_AUGMENTATION_DIFFERENT_EACH_CHANNEL,
                    SAMPLING_STRATEGY_SYSTEMATIC, SAMPLING_STRATEGY_WINDOW)


REGEX_PICKLE = re.compile(
    r"pc_(?P<qrcode>[a-zA-Z0-9]+-[a-zA-Z0-9]+)_(?P<unixepoch>\d+)_(?P<code>\d{3})_(?P<idx>\d{3}).p$"
)


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32),  # (240,180,5)
                              tf.TensorSpec(None, tf.float32),  # (1,)
                              ])
def tf_augment_sample(depthmap, targets):
    depthmap_aug = tf.numpy_function(augment, [depthmap, CONFIG.DATA_AUGMENTATION_MODE], tf.float32)
    depthmap_aug.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, CONFIG.N_ARTIFACTS))
    targets.set_shape((len(CONFIG.TARGET_INDEXES,)))

    return depthmap_aug, targets


def augment(image: np.ndarray, mode=DATA_AUGMENTATION_SAME_PER_CHANNEL) -> np.ndarray:
    assert len(image.shape) == 3, f"image array should have 3 dimensions, but has {len(image.shape)}"
    height, width, n_channels = image.shape
    image = image.astype(np.float32)
    mode = mode.decode("utf-8") if isinstance(mode, bytes) else mode

    if mode == DATA_AUGMENTATION_SAME_PER_CHANNEL:
        # Split channel into separate greyscale images
        # for imgaug this order means: (N, height, width, channels)
        image_reshaped = image.reshape(n_channels, height, width, 1)
        return gen_data_aug_sequence().augment_images(image_reshaped).reshape(height, width, n_channels)

    elif mode == DATA_AUGMENTATION_DIFFERENT_EACH_CHANNEL:
        image_augmented = np.zeros((height, width, n_channels), dtype=np.float32)
        for i in range(n_channels):
            onechannel_img = image[:, :, i]
            image_augmented[:, :, i] = gen_data_aug_sequence().augment_images(onechannel_img).reshape(height, width)
        return image_augmented

    elif mode == DATA_AUGMENTATION_NO:
        return image
    else:
        raise NameError(f"{CONFIG.DATA_AUGMENTATION_MODE}: unknown data aug mode")


def gen_data_aug_sequence():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-10, 10)),
        # brightness  # TODO find out if this makes sense for depthmaps (talk to Lubos)
        sometimes(iaa.Multiply((0.95, 1.1))),
        iaa.CropAndPad(percent=(-0.02, 0.02), pad_cval=(-0.1, 0.1)),  # TODO is this useful for regression on photos?
        iaa.GaussianBlur(sigma=(0, 1.0)),
        sometimes(
            iaa.OneOf(
                [
                    iaa.Dropout((0.01, 0.05)),
                    iaa.CoarseDropout((0.01, 0.03), size_percent=(0.05, 0.1)),
                    iaa.AdditiveGaussianNoise(scale=(0.0, 0.1)),
                    iaa.SaltAndPepper((0.001, 0.005)),
                ]
            ),
        ),
    ])
    return seq


def sometimes(aug):
    """Randomly enable/disable some of the augmentations"""
    return iaa.Sometimes(0.5, aug)


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


def create_multiartifact_paths(qrcode_path: str, n_artifacts: int) -> List[List[str]]:
    """Look at files for 1 qrcode and divide into samples.

    Args:
        qrcode_path: File path of 1 qrcode, e.g. "dataset/scans/1583462470-16tvfmb1d0/100"
        n_artifacts: Desired number of artifacts in one sample

    Returns:
        List of samples, where each sample consists of muliple file paths
    """
    path_with_wildcard = os.path.join(qrcode_path, "*.p")
    list_of_pickle_file_paths = sorted(glob.glob(path_with_wildcard))

    # Split if there are multiple scans on different days
    scans = [list(v) for _unixepoch, v in groupby(list_of_pickle_file_paths, _get_epoch)]

    # Filter to keep scans with enough artifacts
    scans = list(filter(lambda x: len(x) > n_artifacts, scans))

    # Sample artifacts
    if CONFIG.SAMPLING_STRATEGY == SAMPLING_STRATEGY_SYSTEMATIC:
        samples = list(map(partial(sample_systematic_from_artifacts, n_artifacts=n_artifacts), scans))

    if CONFIG.SAMPLING_STRATEGY == SAMPLING_STRATEGY_WINDOW:
        samples = []
        for scan in scans:
            some_samples = list(sample_windows_from_artifacts(scan, n_artifacts=n_artifacts))
            assert len(scan) - n_artifacts + 1 == len(some_samples)
            samples.extend(some_samples)

    return samples


def sample_windows_from_artifacts(artifacts: list, n_artifacts: int) -> Iterator[list]:
    """Sample multiple windows (of length n_artifacts) from list of artifacts

    Args:
        artifacts: e.g. ['001.p', '002.p', '003.p', '004.p', '005.p', '006.p']
        n_artifacts: Desired number of artifacts in one sample

    Returns:
        samples: e.g. [
            ['001.p', '002.p', '003.p', '004.p', '005.p'],
            ['002.p', '003.p', '004.p', '005.p', '006.p'],
        ]
    """
    it = iter(artifacts)
    result = list(islice(it, n_artifacts))
    if len(result) == n_artifacts:
        yield result
    for elem in it:
        result = result[1:] + [elem]
        yield result


def sample_systematic_from_artifacts(artifacts: list, n_artifacts: int) -> list:
    n_artifacts_total = len(artifacts)
    n_skip = n_artifacts_total // n_artifacts  # 20 / 5 = 4
    indexes_to_select = list(range(n_skip // 2, n_artifacts_total, n_skip))[:n_artifacts]
    selected_artifacts = [artifacts[i] for i in indexes_to_select]
    assert len(selected_artifacts) == n_artifacts, str(artifacts)
    return selected_artifacts


def preprocess_targets(targets, targets_indices):
    if targets_indices is not None:
        targets = targets[targets_indices]
    return targets.astype("float32")


def preprocess_depthmap(depthmap):
    # TODO here be more code.
    return depthmap.astype("float32")


def _get_epoch(fname: str) -> str:
    match_result = REGEX_PICKLE.search(fname)
    return match_result.group("unixepoch")
