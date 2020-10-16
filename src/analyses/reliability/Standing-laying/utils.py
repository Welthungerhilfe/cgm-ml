from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

REPO_DIR = Path(__file__).parents[4].absolute()


def get_timestamp_from_pcd(pcd_path):
    filename = str(pcd_path)
    infile = open(filename, 'r')
    try:
        firstLine = infile.readline()
    except Exception as error:
        print(error)
        print(pcd_path)
        return -1
    # get the time from the header of the pcd file
    import re
    timestamp = re.findall(r'\d+\.\d+', firstLine)

    # check if a timestamp is parsed from the header of the pcd file
    try:
        return_timestamp = float(timestamp[0])
    except IndexError:
        return_timestamp = []

    return return_timestamp  # index error? IndexError


def get_timestamp_from_rgb(rgb_path):
    return float(rgb_path[0:-4].split('/')[-1].split('_')[-1])


def find_closest(A, target):
    # A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


def standing_laying_predict(qrcode_pcd_rgb,model):
    qr_codes_predicts = []
    for qr_code in qrcode_pcd_rgb:
        qr_code_predict = []
        for i in tqdm(range(len(qr_code))):
            file = qr_code[i][0]
            img = tf.io.read_file(file)                 # read the image in tensorflow
            img = tf.image.decode_jpeg(img, channels=3)   # change the jpg to rgb
            img = tf.cast(img, tf.float32) * (1. / 256)   # Normalization Not necessary
            if '_100_' in file or '_101_' in file or '_102_' in file:
                img = tf.image.rot90(img, k=3)  # rotate the standing by 270 counter-clockwise
            if '_200_' in file or '_201_' in file or '_202_' in file:
                img = tf.image.rot90(img, k=1)  # rotate the laying by 90 counter-clockwise
            img = tf.image.resize(img, [240, 180])  # Resize the image by 240 * 180
            # Increase the dimesion so that it can fit as a input in model.predict
            img = tf.expand_dims(img, axis=0)
            qr_code_predict.append([model.predict(img), qr_code[i][1], qr_code[i][0]])
        qr_codes_predicts.append(qr_code_predict)

    return qr_codes_predicts
