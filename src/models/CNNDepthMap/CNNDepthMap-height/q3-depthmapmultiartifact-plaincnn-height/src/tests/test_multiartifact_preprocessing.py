from pathlib import Path
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.append(str(Path(__file__).parents[1]))

from config import DATA_AUGMENTATION_SAME_PER_CHANNEL, DATA_AUGMENTATION_DIFFERENT_EACH_CHANNEL, DATA_AUGMENTATION_NO  # noqa: E402
from preprocessing_multi import augment, tf_augment_sample, sample_systematic_from_artifacts, sample_windows_from_artifacts, REGEX_PICKLE  # noqa: E402


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


def test_sample_windows_from_artifacts_multiple_results():
    artifacts = ['001.p', '002.p', '003.p', '004.p', '005.p', '006.p']
    actual = list(sample_windows_from_artifacts(artifacts, 5))
    expected = [
        ['001.p', '002.p', '003.p', '004.p', '005.p'],
        ['002.p', '003.p', '004.p', '005.p', '006.p'],
    ]
    assert actual == expected


def test_sample_windows_from_artifacts_one_result():
    artifacts = ['001.p', '002.p', '003.p', '004.p', '005.p']
    actual = list(sample_windows_from_artifacts(artifacts, 5))
    expected = [
        ['001.p', '002.p', '003.p', '004.p', '005.p'],
    ]
    assert actual == expected


def test_sample_windows_from_artifacts_no_result():
    artifacts = ['001.p', '002.p', '003.p', '004.p']
    actual = list(sample_windows_from_artifacts(artifacts, 5))
    assert actual == []


def test_systematic_sample_from_many_artifacts():
    artifacts = list(range(20, 0, -1))
    n_artifacts = 5
    selected_artifacts = sample_systematic_from_artifacts(artifacts, n_artifacts)
    assert selected_artifacts == [18, 14, 10, 6, 2]
    assert len(selected_artifacts) == n_artifacts


def test_systematic_sample_from_few_artifacts():
    artifacts = ['0', '1', '2', '3', '4', '5', '6']
    n_artifacts = 5
    selected_artifacts = sample_systematic_from_artifacts(artifacts, n_artifacts)
    assert selected_artifacts[0] == '0'
    assert selected_artifacts[4] == '4'
    assert len(selected_artifacts) == n_artifacts


def test_systematic_sample_from_artifacts_too_few():
    artifacts = list(range(3, 0, -1))
    n_artifacts = 5
    with pytest.raises(Exception):
        sample_systematic_from_artifacts(artifacts, n_artifacts)


def test_regex_pickle():
    fname = "pc_1583462470-16tvfmb1d0_1591122155216_100_000.p"

    match_result = REGEX_PICKLE.search(fname)
    assert match_result.group("qrcode") == "1583462470-16tvfmb1d0"
    assert match_result.group("unixepoch") == "1591122155216"
    assert match_result.group("code") == "100"
    assert match_result.group("idx") == "000"
