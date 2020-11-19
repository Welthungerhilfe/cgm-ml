import os
from pathlib import Path
import pickle
import tensorflow as tf

import pandas as pd

REPO_DIR = Path(os.getcwd()).parents[2]

# Error margin on various ranges
EVALUATION_ACCURACIES = [.2, .4, .8, 1.2, 2., 2.5, 3., 4., 5., 6.]
MODEL_CKPT_FILENAME = "best_model.ckpt"

CODE_TO_SCANTYPE = {
    '100': '_front',
    '101': '_360',
    '102': '_back',
    '200': '_lyingfront',
    '201': '_lyingrot',
    '202': '_lyingback',
}


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


CONFIG = dotdict(dict(
    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,
    NORMALIZATION_VALUE=7.5,
    TARGET_INDEXES=[0],  # 0 is height, 1 is weight.
    N_ARTIFACTS=5,
    CODES_FOR_POSE_AND_SCANSTEP=("100", ),
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
        # print(f"Accuracy {acc:.1f} for {code}: {accuracy}")
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
    if targets_indices is not None:
        targets = targets[targets_indices]
    return targets.astype("float32")


def preprocess_depthmap(depthmap):
    # TODO here be more code.
    return depthmap.astype("float32")
