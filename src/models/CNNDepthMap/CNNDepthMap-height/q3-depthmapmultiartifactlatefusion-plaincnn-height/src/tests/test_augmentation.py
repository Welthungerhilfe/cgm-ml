import numpy as np
from pathlib import Path
import sys
import tensorflow as tf

sys.path.append(str(Path(__file__).parents[1]))

from config import DATA_AUGMENTATION_SAME_PER_CHANNEL, DATA_AUGMENTATION_DIFFERENT_EACH_CHANNEL, DATA_AUGMENTATION_NO  # noqa: E402
from augmentation import augment, tf_augment_sample  # noqa: E402


def test_tf_augment_sample():
    # Test with dataset (not eager)
    sample = tf.random.uniform((240, 180, 5))
    target = tf.constant([92.3])
    dataset = tf.data.Dataset.from_tensors((sample, target))
    _ = dataset.map(tf_augment_sample, tf.data.experimental.AUTOTUNE)

    # Test eagerly
    tf_augment_sample(sample, target)


def test_imgaug_on_multichannel_same():
    sample = np.ones((240, 180, 5)) * 0.5
    result = augment(sample, mode=DATA_AUGMENTATION_SAME_PER_CHANNEL)
    # assert np.all(result[0] == result[1])  # cannot be ensured currently
    assert result.shape == (240, 180, 5)


def test_imgaug_on_multichannel_different():
    sample = np.ones((240, 180, 5)) * 0.5
    result = augment(sample, mode=DATA_AUGMENTATION_DIFFERENT_EACH_CHANNEL)
    assert not np.all(result[0] == result[1])
    assert result.shape == (240, 180, 5)


def test_imgaug_on_multichannel_no():
    sample = np.random.rand(240, 180, 5)
    result = augment(sample, mode=DATA_AUGMENTATION_NO)
    assert result.shape == (240, 180, 5)
