import datetime
import os
import pickle
import random
import re
from pathlib import Path

import logging
import logging.config

import glob2 as glob
import matplotlib.pyplot as plt
import tensorflow as tf

from common.model_utils.preprocessing import preprocess_depthmap, preprocess_targets

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

REPO_DIR = Path(os.getcwd()).parent

IMAGE_TARGET_HEIGHT = 240
IMAGE_TARGET_WIDTH = 180
TARGET_INDEXES = [0]  # 0 is height, 1 is weight.

REGEX_PICKLE = re.compile(
    r"pc_(?P<qrcode>[a-zA-Z0-9]+-[a-zA-Z0-9]+)_(?P<unixepoch>\d+)_(?P<code>\d{3})_(?P<idx>\d{3}).p"
)


def py_load_pickle(path, max_value=7.5):
    path_ = path if isinstance(path, str) else path.numpy()
    depthmap, targets = pickle.load(open(path_, "rb"))
    depthmap = preprocess_depthmap(depthmap)
    depthmap = depthmap / max_value
    depthmap = tf.image.resize(depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
    targets = preprocess_targets(targets, TARGET_INDEXES)
    return depthmap, targets


def path_to_ndarray(pickle_file_path):
    depthmap, _targets = py_load_pickle(pickle_file_path)
    depthmap = tf.reshape(depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
    return depthmap.numpy(), _targets


def show_pickle(pickle_file_path):
    depthmap, _targets = py_load_pickle(pickle_file_path)
    depthmap = tf.reshape(depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
    plt.imshow(depthmap.numpy(), cmap='gray', vmin=0, vmax=1)
    logging.info('Height: %s cm', _targets[0])


def _get_epoch(fpath: str) -> str:
    fname = os.path.basename(fpath)
    match_result = REGEX_PICKLE.search(fname)
    return match_result.group("unixepoch")


def get_datetime(fpath: str):
    epoch = _get_epoch(fpath)
    return str(datetime.datetime.fromtimestamp(int(epoch[:-3])))


def choose_a_pickle_file(qrcodes_with_wildcard, scans_dir):
    len_pickle_paths = 0
    while len_pickle_paths == 0:
        qrcode = random.choice(qrcodes_with_wildcard)
        artifacts_dir = scans_dir / qrcode / "100"
        path_with_wildcard = os.path.join(artifacts_dir, "*.p")
        list_of_pickle_file_paths = glob.glob(path_with_wildcard)
        len_pickle_paths = len(list_of_pickle_file_paths)
    pickle_file_path = random.choice(list_of_pickle_file_paths)
    return pickle_file_path
