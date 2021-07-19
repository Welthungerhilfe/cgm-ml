import os
import logging
import logging.config
from pathlib import Path
import pickle
import tensorflow as tf

from bunch import Bunch
import pandas as pd

from .constants_eval import (  # noqa: E402, F401
    AGE_IDX, COLUMN_NAME_AGE, COLUMN_NAME_GOODBAD, COLUMN_NAME_SEX, CONFIG, EVALUATION_ACCURACIES,
    GOODBAD_DICT, GOODBAD_IDX, HEIGHT_IDX, SEX_DICT, SEX_IDX, WEIGHT_IDX)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

REPO_DIR = Path(os.getcwd()).parents[2]


def calculate_performance(code: str,
                          df_mae: pd.DataFrame,
                          result_config: Bunch) -> pd.DataFrame:
    """For a specific scantype, calculate the performance of the model on each error margin
    Args:
        code: e.g. '100'
        df_mae: dataframe
        result_config: bunch containing result config
    Returns:
        dataframe, where each column describes a differnt accuracy, e.g.
                            0.2   0.4   0.6   1.0   1.2    2.0    2.5    3.0    4.0    5.0    6.0
                           20.0  20.0  40.0  80.0  80.0  100.0  100.0  100.0  100.0  100.0  100.0
    """
    df_mae_filtered = df_mae.iloc[df_mae.index.get_level_values('scantype') == code]
    accuracy_list = []
    for acc in result_config.ACCURACIES:
        good_predictions = df_mae_filtered[(df_mae_filtered['error'] <= acc) & (df_mae_filtered['error'] >= -acc)]
        if len(df_mae_filtered) > 0:
            accuracy = len(good_predictions) / len(df_mae_filtered) * 100
        else:
            accuracy = 0.
        accuracy_list.append(accuracy)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T
    df_out.columns = result_config.ACCURACIES
    return df_out


# Function for loading and processing depthmaps.
def tf_load_pickle(path, max_value):
    def py_load_pickle(path, max_value):
        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = preprocess_depthmap(depthmap)
        depthmap = depthmap / max_value
        depthmap = tf.image.resize(depthmap, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH))
        targets = preprocess_targets(targets, CONFIG.TARGET_INDEXES)
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
    depthmap.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1))
    targets.set_shape((len(CONFIG.TARGET_INDEXES,)))
    return depthmap, targets


def extract_qrcode(row):
    qrc = row['artifacts'].split('/')[-3]
    return qrc


def extract_scantype(row):
    """https://dev.azure.com/cgmorg/ChildGrowthMonitor/_wiki/wikis/ChildGrowthMonitor.wiki/15/Codes-for-Pose-and-Scan-step"""  # noqa: E501
    scans = row['artifacts'].split('/')[-2]
    return scans


def avgerror(row):
    difference = row['GT'] - row['predicted']
    return difference


def preprocess(path):
    depthmap, targets = pickle.load(open(path, "rb"))
    depthmap = preprocess_depthmap(depthmap)
    depthmap = depthmap / CONFIG.NORMALIZATION_VALUE
    depthmap = tf.image.resize(depthmap, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH))
    targets = preprocess_targets(targets, CONFIG.TARGET_INDEXES)
    depthmap.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 1))
    return depthmap, targets


def preprocess_targets(targets, targets_indices):
    if SEX_IDX in targets_indices:
        targets[SEX_IDX] = SEX_DICT[targets[SEX_IDX]]
    if GOODBAD_IDX in targets_indices:
        try:
            targets[GOODBAD_IDX] = GOODBAD_DICT[targets[GOODBAD_IDX]]
        except KeyError:
            logging.info("Key %s not found in GOODBAD_DICT", targets[GOODBAD_IDX])
            targets[GOODBAD_IDX] = GOODBAD_DICT['delete']  # unknown target values will be categorized as 'delete'

    if targets_indices is not None:
        targets = targets[targets_indices]
    return targets.astype("float32")


def preprocess_depthmap(depthmap):
    return depthmap.astype("float32")
