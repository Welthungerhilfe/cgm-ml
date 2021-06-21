#from src.common.evaluation.QA.eval_depthmap_models.src.evaluate import prepare_sample_dataset
#from src.common.evaluation.QA.eval_depthmap_models.src.evaluate import tf_load_pickle
import sys
import shutil
from pathlib import Path

import pandas as pd
import pickle

here = Path(__file__).resolve()  # not sure if this is best practice - find out

sys.path.append(str(here.parents[4]))

REPO_DIR = str(here.parents[6].absolute())

from evaluate import copy_dir, prepare_sample_dataset, tf_load_pickle  # noqa: E402


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
    # prepare sample dataset
    dataset_path = str(REPO_DIR + "/data/anon-depthmap-mini/scans")

    #prepare df
    qrcode_list = [
        "1585004725-18cqo1np0j",
        "1585004725-18cqo1np0j",
        "1585012629-ac1ippx2qy",
        "1585012629-ac1ippx2qy",
        "1585012645-ii2eyvdpib",
        "1585012645-ii2eyvdpib",
    ]

    scantype_list = [
        "100",  # 1st qrcode
        "102",
        "100",  # 2nd qrcode
        "102",
        "100",  # 3rd qrcode
        "102",
    ]

    artifact_list = [
        "pc_1585004725-18cqo1np0j_1592801845251_100_000.p",  # 100 1st qrcode
        "pc_1585004725-18cqo1np0j_1592801845251_102_000.p",  # 102
        "pc_1585012629-ac1ippx2qy_1591848606827_100_000.p",  # 100 2nd qrcode
        "pc_1585012629-ac1ippx2qy_1591848606827_102_072.p",  # 102
        "pc_1585012645-ii2eyvdpib_1591848619397_100_000.p",  # 100 3rd qrcode
        "pc_1585012645-ii2eyvdpib_1591848619397_102_046.p",  # 102
    ]

    prediction_list = [
        96.8,
        96.8,
        85.3,
        84.8,
        79.1,
        79.6,
    ]

    target_list = [
        95.5,
        95.5,
        85.0,
        85.0,
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

    print("Exiting test_prepare_sample_dataset")

    assert len(dataset_sample), 'There should at least be 1 dataset sample'


#tf load pickle
def test_tf_load_pickle():
    pickle_path = str(REPO_DIR
                      + "/data/anon-depthmap-mini/scans/1585004725-18cqo1np0j/100/"
                      + "pc_1585004725-18cqo1np0j_1592801845251_100_000.p")

    NORMALIZATION_VALUE = 7.5
    IMAGE_TARGET_HEIGHT = 240
    IMAGE_TARGET_WIDTH = 180

    test_image = tf_load_pickle(pickle_path, NORMALIZATION_VALUE)

    print(type(test_image))
    print("test_image[0] = ", test_image[0])  # path
    print("test_image[1] = ", test_image[1])  # pixel data
    print("test_image[1] shape 0 = ", test_image[1].shape[0])  # img height 240
    print("test_image[1] shape 1 = ", test_image[1].shape[1])  # img width 180
    print("test_image[1] shape 2 = ", test_image[1].shape[2])
    print("test_image[2] = ", test_image[2])  # tensor info

    assert (type(test_image)
            == tuple) and (test_image[1].shape[0]
                           == IMAGE_TARGET_HEIGHT) and (test_image[1].shape[1]
                                                        == IMAGE_TARGET_WIDTH)


# not sure if this is good
def test_tf_load_not_a_pickle():
    exception_caught = False
    try:
        wrong_path = str(REPO_DIR + "/data/anon-depthmap-mini/labels/testing.csv")

        NORMALIZATION_VALUE = 7.5

        test_image = tf_load_pickle(wrong_path, NORMALIZATION_VALUE)

        print(type(test_image))
        print("test_image[0] = ", test_image[0])  # path
        print("test_image[1] = ", test_image[1])  # pixel data
        print("test_image[1] shape 0 = ", test_image[1].shape[0])  # img height 240
        print("test_image[1] shape 1 = ", test_image[1].shape[1])  # img width 180
        print("test_image[1] shape 2 = ", test_image[1].shape[2])
        print("test_image[2] = ", test_image[2])  # tensor info
    except Exception:
        print("exception: Unknown: UnpicklingError")
        exception_caught = True

    assert(exception_caught)


def test_tf_load_empty_pickle():
    exception_caught = False
    try:
        # create empty list to dump into pickle file
        print("hey")
        empty_dict = {}
        empty_file = open('empty_pickle_file.p', 'wb')
        pickle.dump(empty_dict, empty_file)
        print(empty_file)
        print(type(empty_file))
        empty_file.close()
        print("ho")

        NORMALIZATION_VALUE = 7.5
        test_image = tf_load_pickle('empty_pickle_file.p', NORMALIZATION_VALUE)

        print(type(test_image))
        print("test_image[0] = ", test_image[0])  # path
        print("test_image[1] = ", test_image[1])  # pixel data
        print("test_image[1] shape 0 = ", test_image[1].shape[0])  # img height 240
        print("test_image[1] shape 1 = ", test_image[1].shape[1])  # img width 180
        print("test_image[1] shape 2 = ", test_image[1].shape[2])
        print("test_image[2] = ", test_image[2])  # tensor info
    except Exception:
        print("value error")
        exception_caught = True

    assert(exception_caught)


def test_always_passes():
    assert True


"""def test_always_fails():
    assert False"""
