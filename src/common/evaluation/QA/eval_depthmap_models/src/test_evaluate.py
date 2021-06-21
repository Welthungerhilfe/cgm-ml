#from src.common.evaluation.QA.eval_depthmap_models.src.evaluate import prepare_sample_dataset
import sys
import shutil
from pathlib import Path

import pandas as pd

here = Path(__file__).resolve()  # not sure if this is best practice - find out

sys.path.append(str(here.parents[4]))

REPO_DIR = str(here.parents[6].absolute())

from evaluate import copy_dir  # noqa: E402
from evaluate import prepare_sample_dataset  # noqa: E402


def test_copy_dir():
    common_dir_path = Path(REPO_DIR + "/src/common")
    temp_common_dir = Path(here.parent / "temp_common")
    copy_dir(src=common_dir_path, tgt=temp_common_dir, glob_pattern='*/*.py', should_touch_init=True)

    assert temp_common_dir.exists()
    try:
        shutil.rmtree(temp_common_dir)
    except OSError as e:
        print("Error: %s : %s" % (temp_common_dir, e.strerror))


#include empty dir
def test_copy_empty_dir():
    #create empty dir
    empty_path = Path(here.parent / "copy_empty")
    empty_path.mkdir(parents=True, exist_ok=True)
    #copy empty dir into temp_empty_dir
    temp_empty_dir = Path(here.parent / "temp_empty_dir")
    copy_dir(src=empty_path, tgt=temp_empty_dir, glob_pattern='*/*.py', should_touch_init=False)

    assert temp_empty_dir.exists()

    try:
        shutil.rmtree(empty_path)
    except OSError as e:
        print("Error: %s : %s" % (empty_path, e.strerror))

    try:
        shutil.rmtree(temp_empty_dir)
    except OSError as e:
        print("Error: %s : %s" % (temp_empty_dir, e.strerror))


def test_prepare_sample_dataset():
    print("Entering test_prepare_sample_dataset")
    # prepare sample dataset
    dataset_path = str(REPO_DIR + "/data/anon-depthmap-mini")
    print("datasetpath", dataset_path)

    #prepare df
    qrcode_list = [
        "1585004725-18cqo1np0j",
        "1585004725-18cqo1np0j",
        "1585004725-18cqo1np0j",
        "1585004725-18cqo1np0j",
        "1585004725-18cqo1np0j",
        "1585012629-ac1ippx2qy",
        "1585012629-ac1ippx2qy",
        "1585012629-ac1ippx2qy",
        "1585012629-ac1ippx2qy",
        "1585012629-ac1ippx2qy",
        "1585012645-ii2eyvdpib",
        "1585012645-ii2eyvdpib",
        "1585012645-ii2eyvdpib",
        "1585012645-ii2eyvdpib",
        "1585012645-ii2eyvdpib",
    ]

    scantype_list = [
        "100",  # 1st qrcode
        "100",
        "100",
        "101",
        "102",
        "100",  # 2nd qrcode
        "101",
        "101",
        "102",
        "102",
        "100",  # 3rd qrcode
        "100",
        "101",
        "102",
        "102",
    ]

    artifact_list = [
        "pc_1585004725-18cqo1np0j_1592801845251_100_000.p",  # 100 1st qrcode
        "pc_1585004725-18cqo1np0j_1592801845251_100_001.p",  # 100
        "pc_1585004725-18cqo1np0j_1592801845251_100_002.p",  # 100
        "pc_1585004725-18cqo1np0j_1592801845251_101_000.p",  # 101
        "pc_1585004725-18cqo1np0j_1592801845251_102_000.p",  # 102
        "pc_1585012629-ac1ippx2qy_1591848606827_100_000.p",  # 100 2nd qrcode
        "pc_1585012629-ac1ippx2qy_1591848606827_101_014.p",  # 101
        "pc_1585012629-ac1ippx2qy_1591848606827_101_015.p",  # 101
        "pc_1585012629-ac1ippx2qy_1591848606827_102_071.p",  # 102
        "pc_1585012629-ac1ippx2qy_1591848606827_102_072.p",  # 102
        "pc_1585012645-ii2eyvdpib_1591848619397_100_000.p",  # 100 3rd qrcode
        "pc_1585012645-ii2eyvdpib_1591848619397_100_002.p",  # 100
        "pc_1585012645-ii2eyvdpib_1591848619397_101_008.p",  # 101
        "pc_1585012645-ii2eyvdpib_1591848619397_102_045.p",  # 102
        "pc_1585012645-ii2eyvdpib_1591848619397_102_046.p",  # 102
    ]

    prediction_list = [
        96.8,
        96.0,
        95.8,
        94.5,
        96.8,
        85.3,
        84.9,
        85.0,
        85.2,
        84.8,
        79.1,
        78.8,
        79.0,
        79.3,
        79.6,
    ]

    target_list = [
        95.5,
        95.5,
        95.5,
        95.5,
        95.5,
        85.0,
        85.0,
        85.0,
        85.0,
        85.0,
        79.1,
        79.1,
        79.1,
        79.1,
        79.1,
    ]

    COLUMNS = ['qrcode', 'artifact', 'scantype', 'GT', 'predicted']

    df = pd.DataFrame({
        'qrcode': qrcode_list,
        'artifact': artifact_list,
        'scantype': scantype_list,
        'GT': target_list,
        'predicted': prediction_list
    }, columns=COLUMNS)

    df_sample = df.groupby(['qrcode', 'scantype']).apply(lambda x: x.sample(1))

    dataset_sample = prepare_sample_dataset(df_sample, dataset_path)
    print(dataset_sample)

    print("Exiting test_prepare_sample_dataset")

    assert (len(dataset_sample) != 0)  # how can I write a better assert statement here?


def test_always_passes():
    assert True


"""def test_always_fails():
    assert False"""
