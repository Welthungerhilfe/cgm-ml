import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from azureml.core import Experiment, Run
from tqdm import tqdm

REPO_DIR = Path(__file__).parents[4].absolute()

scan_type = {
    'Standing_front': '_100_',
    'Standing_360': '_101',
    'Standing_back': '_102_',
    'Laying_front': '_200_',
    'Laying_360': '_201',
    'Laying_back': '_202_'
}


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
    timestamp = re.findall(r'\d+\.\d+', firstLine)

    # check if a timestamp is parsed from the header of the pcd file
    try:
        return_timestamp = float(timestamp[0])
    except IndexError:
        return_timestamp = []

    return return_timestamp


def get_timestamp_from_rgb(rgb_path):
    return float(rgb_path[0:-4].split('/')[-1].split('_')[-1])


def find_closest(rgb_list: list, target: float) -> int:
    idx = rgb_list.searchsorted(target)
    idx = np.clip(idx, 1, len(rgb_list) - 1)
    left = rgb_list[idx - 1]
    right = rgb_list[idx]
    idx -= target - left < right - target
    return idx


def standing_laying_predict(qrcode_pcd_rgb, model):
    qr_codes_predicts = []
    for qr_code in qrcode_pcd_rgb:
        qr_code_predict = []
        for i in tqdm(range(len(qr_code))):
            file = qr_code[i][0]
            img = tf.io.read_file(file)                 # read the image in tensorflow
            img = tf.image.decode_jpeg(img, channels=3)   # change the jpg to rgb
            img = tf.cast(img, tf.float32) * (1. / 256)   # Normalization Not necessary
            if scan_type['Standing_front'] in file or scan_type['Standing_back'] in file or scan_type['Standing_360'] in file:
                img = tf.image.rot90(img, k=3)  # rotate the standing by 270 counter-clockwise
            if scan_type['Laying_front'] in file or scan_type['Laying_back'] in file or scan_type['Laying_360'] in file:
                img = tf.image.rot90(img, k=1)  # rotate the laying by 90 counter-clockwise
            img = tf.image.resize(img, [240, 180])  # Resize the image by 240 * 180
            # Increase the dimesion so that it can fit as a input in model.predict
            img = tf.expand_dims(img, axis=0)
            qr_code_predict.append([model.predict(img), qr_code[i][1], qr_code[i][0]])
        qr_codes_predicts.append(qr_code_predict)

    return qr_codes_predicts


def download_model(ws, experiment_name, run_id, input_location, output_location):
    '''
    Download the pretrained model
    Input:
         ws: workspace to access the experiment
         experiment_name: Name of the experiment in which model is saved
         run_id: Run Id of the experiment in which model is pre-trained
         input_location: Input location in a RUN Id
         output_location: Location for saving the model
    '''
    experiment = Experiment(workspace=ws, name=experiment_name)
    #Download the model on which evaluation need to be done
    run = Run(experiment, run_id=run_id)
    #run.get_details()
    run.download_file(input_location, output_location)
    print("Successfully downloaded model")
