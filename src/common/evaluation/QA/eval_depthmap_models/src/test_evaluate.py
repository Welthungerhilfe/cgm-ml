import sys
import shutil
from pathlib import Path
import pytest

import pandas as pd

CWD = Path(__file__).resolve()

sys.path.append(str(CWD.parents[4]))

REPO_DIR = str(CWD.parents[6].absolute())

from evaluation.QA.eval_depthmap_models.src.evaluate import (copy_dir, prepare_sample_dataset,  # noqa: E402
                                                             tf_load_pickle)


@pytest.fixture
def temp_common_dir():
    temp_common_dir = Path(CWD.parent / "temp_common")
    yield temp_common_dir
    try:
        shutil.rmtree(temp_common_dir)
    except OSError:
        pass


@pytest.fixture
def empty_dir():
    empty_dir = Path(CWD.parent / "copy_empty")
    empty_dir.mkdir(parents=True, exist_ok=True)
    yield empty_dir
    try:
        shutil.rmtree(empty_dir)
    except OSError:
        pass


@pytest.fixture
def temp_empty_dir():
    temp_empty_dir = Path(CWD.parent / "temp_empty_dir")
    yield temp_empty_dir
    try:
        shutil.rmtree(temp_empty_dir)
    except OSError:
        pass


def test_copy_dir(temp_common_dir):
    common_dir_path = Path(REPO_DIR + "/src/common")
    copy_dir(src=common_dir_path, tgt=temp_common_dir, glob_pattern='*/*.py', should_touch_init=True)
    assert temp_common_dir.is_dir(), 'The temp_common_dir does not exist. Did copy_dir fail?'


def test_copy_empty_dir(empty_dir, temp_empty_dir):
    copy_dir(src=empty_dir, tgt=temp_empty_dir, glob_pattern='*/*.py', should_touch_init=False)
    assert temp_empty_dir.is_dir(), 'The temp_empty_dir does not exist. Did copy_dir fail for an empty directory?'


def test_prepare_sample_dataset():
    dataset_path = str("testfiles")

    qrcode_list = [
        "1585004725-18cqo1np0j",
        "1585004725-18cqo1np0j",
        "1585012629-ac1ippx2qy",
        "1585012629-ac1ippx2qy",
    ]

    scantype_list = [
        "100",
        "100",
        "100",
        "102",
    ]

    artifact_list = [
        "pc_1585004725-18cqo1np0j_1592801845251_100_000.p",
        "pc_1585004725-18cqo1np0j_1592801845251_100_001.p",
        "pc_1585012629-ac1ippx2qy_1591848606827_100_000.p",
        "pc_1585012629-ac1ippx2qy_1591848606827_102_072.p",
    ]

    prediction_list = [
        96.8,
        96.8,
        85.3,
        84.8,
    ]

    target_list = [
        95.5,
        95.5,
        85.0,
        85.0,
    ]

    columns = ['qrcode', 'artifact', 'scantype', 'GT', 'predicted']

    df = pd.DataFrame({
        'qrcode': qrcode_list,
        'artifact': artifact_list,
        'scantype': scantype_list,
        'GT': target_list,
        'predicted': prediction_list
    }, columns=columns)

    df_sample = df.groupby(['qrcode', 'scantype']).apply(lambda x: x.sample(1))
    dataset_sample = prepare_sample_dataset(df_sample, dataset_path)

    assert len(dataset_sample) == 3, 'There should be 3 samples in the dataset'


def test_tf_load_pickle():
    pickle_path = str(REPO_DIR
                      + "/src/common/data_utilities/tests/pickle_files/scans/c571de02-"
                      + "a723-11eb-8845-bb6589a1fbe8/102/pc_c571de02-a723-11eb-8845-bb"
                      + "6589a1fbe8_2021-04-22 13:34:33.302557_102_3.p")

    normalization_value = 7.5
    image_target_height = 240
    image_target_width = 180
    test_image = tf_load_pickle(pickle_path, normalization_value)

    assert isinstance(test_image, tuple), 'The type of object should be a tuple'
    assert test_image[1].shape[0] == image_target_height, f"The height of the object should be {image_target_height}"
    assert test_image[1].shape[1] == image_target_width, f"The width of the object should be {image_target_width}"


def test_tf_load_not_a_pickle():
    wrong_path = str(REPO_DIR + "/src/common/data_utilities/tests/zip_files/be1faf54-"
                     + "69c7-11eb-984b-a3ffd42e7b5a/depth/bd67cd9e-69c7-11eb-984b-77ac9d2b4986")
    normalization_value = 7.5
    with pytest.raises(Exception, match='UnpicklingError'):
        tf_load_pickle(wrong_path, normalization_value)


def test_tf_load_empty_pickle():
    empty_path = str(REPO_DIR + "/src/common/evaluation/QA/eval_depthmap_models/src/testfiles/empty_pickle_file.p")
    normalization_value = 7.5
    with pytest.raises(Exception, match='not enough values to unpack'):
        tf_load_pickle(empty_path, normalization_value)
