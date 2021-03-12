import os
import logging
import logging.config
from pathlib import Path
import pickle
import tensorflow as tf

from bunch import Bunch
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

REPO_DIR = Path(os.getcwd()).parents[2]

# Error margin on various ranges
EVALUATION_ACCURACIES = [.2, .4, .6, 1., 1.2, 2., 2.5, 3., 4., 5., 6.]
MODEL_CKPT_FILENAME = "best_model.ckpt"

DAYS_IN_YEAR = 365

HEIGHT_IDX = 0
WEIGHT_IDX = 1
MUAC_IDX = 2
AGE_IDX = 3
SEX_IDX = 4
GOODBAD_IDX = 5

SEX_DICT = {'female': 0., 'male': 1.}
GOODBAD_DICT = {'bad': 0., 'good': 1., 'delete': 2.}

COLUMN_NAME_AGE = 'GT_age'
COLUMN_NAME_SEX = 'GT_sex'
COLUMN_NAME_GOODBAD = 'GT_goodbad'
CODE_TO_SCANTYPE = {
    '100': '_standingfront',
    '101': '_standing360',
    '102': '_standingback',
    '200': '_lyingfront',
    '201': '_lyingrot',
    '202': '_lyingback',
}

DATA_AUGMENTATION_SAME_PER_CHANNEL = "same_per_channel"
DATA_AUGMENTATION_DIFFERENT_EACH_CHANNEL = "different_each_channel"
DATA_AUGMENTATION_NO = "no"

SAMPLING_STRATEGY_SYSTEMATIC = "systematic"
SAMPLING_STRATEGY_WINDOW = "window"

CONFIG = Bunch(dict(
    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,
    NORMALIZATION_VALUE=7.5,
    TARGET_INDEXES=[0],  # 0 is height, 1 is weight.
    DATA_AUGMENTATION_MODE=DATA_AUGMENTATION_NO,
    SAMPLING_STRATEGY=SAMPLING_STRATEGY_SYSTEMATIC,
    N_ARTIFACTS=5,
    N_REPEAT_DATASET=1,
    CODES_FOR_POSE_AND_SCANSTEP=('100', '101', '102', '200', '201', '202'),
))


def calculate_performance(code, df_mae):
    df_mae_filtered = df_mae.iloc[df_mae.index.get_level_values('scantype') == code]
    accuracy_list = []
    for acc in EVALUATION_ACCURACIES:
        good_predictions = df_mae_filtered[(df_mae_filtered['error'] <= acc) & (df_mae_filtered['error'] >= -acc)]
        if len(df_mae_filtered):
            accuracy = len(good_predictions) / len(df_mae_filtered) * 100
        else:
            accuracy = 0.
        # logging.info("Accuracy %d for code %s: %d", acc, code, accuracy)
        accuracy_list.append(accuracy)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T
    df_out.columns = EVALUATION_ACCURACIES
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
    """https://dev.azure.com/cgmorg/ChildGrowthMonitor/_wiki/wikis/ChildGrowthMonitor.wiki/15/Codes-for-Pose-and-Scan-step"""
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
    # TODO here be more code.
    return depthmap.astype("float32")
