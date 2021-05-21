import math
import os
import logging
import logging.config
import pickle
from pathlib import Path
import sys
import zipfile
from typing import Tuple

from azureml.core import Experiment, Run
import glob2 as glob
import numpy as np
import pandas as pd
from skimage.transform import resize

sys.path.append(str(Path(__file__).parents[0]))

from qa_config import DATA_CONFIG, EVAL_CONFIG, RESULT_CONFIG  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

image_target_height = 240
image_target_width = 180

measure_name_list = ['HEIGHT', 'WEIGHT', 'MUAC']
measure_category_list = ['GOOD', 'ACCEPTABLE', 'POOR', 'REJECT']
permissible_measure = {'HEIGHT':
                       {'GOOD': 0.4, 'ACCEPTABLE': 0.6, 'POOR': 1.0, 'REJECT': None},
                       'WEIGHT':
                       {'GOOD': 0.04, 'ACCEPTABLE': 0.10, 'POOR': 0.21, 'REJECT': None},
                       'MUAC':
                       {'GOOD': 2.0, 'ACCEPTABLE': 2.7, 'POOR': 3.3, 'REJECT': None}
                       }

CODE_TO_SCANTYPE = {
    '100': '_standingfront',
    '101': '_standing360',
    '102': '_standingback',
    '200': '_lyingfront',
    '201': '_lyingrot',
    '202': '_lyingback',
}


def get_intra_TEM(measure_one, measure_two):
    '''
    https://www.scielo.br/pdf/rbme/v11n1/en_24109.pdf
    Compute Intra Technical Error of Measurement
    '''
    assert(len(measure_one.index) == len(measure_two.index))
    sum_of_square_of_deviation = ((measure_one - measure_two) ** 2).sum()
    absolute_TEM = math.sqrt(sum_of_square_of_deviation / (2 * len(measure_one.index)))

    if EVAL_CONFIG.DEBUG_LOG:
        logging.info("Absolute TEM : %d", absolute_TEM)

    return absolute_TEM


def get_measure_category(technical_error_of_measurement, measure_name):
    '''
    Return the measure category based on the technical error of measurement
    e.g. GOOD, ACCEPTABLE, POOR, REJECT
    '''
    if technical_error_of_measurement < permissible_measure[measure_name]['GOOD']:
        measure_category = 'GOOD'
    elif technical_error_of_measurement < permissible_measure[measure_name]['ACCEPTABLE']:
        measure_category = 'ACCEPTABLE'
    elif technical_error_of_measurement < permissible_measure[measure_name]['POOR']:
        measure_category = 'POOR'
    else:
        measure_category = 'REJECT'

    return measure_category


def preprocess_depthmap(depthmap):
    # TODO here be more code.
    return depthmap.astype("float32")


def preprocess_targets(targets, targets_indices):
    if targets_indices is not None:
        targets = targets[targets_indices]
    return targets.astype("float32")


def get_depthmap_files(paths):
    pickle_paths = []
    for path in paths:
        pickle_paths.extend(glob.glob(os.path.join(path, "**", "*.p")))
    return pickle_paths


def get_column_list(depthmap_path_list, prediction):
    '''
    Prepare the columns to be included in the artifact level dataframe
    '''
    qrcode_list, scan_type_list, artifact_list, prediction_list, target_list = [], [], [], [], []

    for idx, path in enumerate(depthmap_path_list):
        _, targets = pickle.load(open(path, "rb"))
        targets = preprocess_targets(targets, DATA_CONFIG.TARGET_INDEXES)
        target = np.squeeze(targets)

        sub_folder_list = path.split('/')
        qrcode_list.append(sub_folder_list[-3])
        scan_type_list.append(sub_folder_list[-2])
        artifact_list.append(sub_folder_list[-1])
        prediction_list.append(prediction[idx])
        target_list.append(target)

    return qrcode_list, scan_type_list, artifact_list, prediction_list, target_list


def avgerror(row):
    difference = row['GT'] - row['predicted']
    return difference


def calculate_performance(code, df_mae):
    df_mae_filtered = df_mae.iloc[df_mae.index.get_level_values('scantype') == code]
    accuracy_list = []
    for acc in RESULT_CONFIG.ACCURACIES:
        good_predictions = df_mae_filtered[(df_mae_filtered['error'] <= acc) & (df_mae_filtered['error'] >= -acc)]
        if len(df_mae_filtered):
            accuracy = len(good_predictions) / len(df_mae_filtered) * 100
        else:
            accuracy = 0.
        # logging.info("Accuracy %d for %d: %d", acc, code, accuracy)
        accuracy_list.append(accuracy)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T
    df_out.columns = RESULT_CONFIG.ACCURACIES
    return df_out


def calculate_and_save_results(MAE, complete_name, CSV_OUT_PATH):
    '''
    Calculate accuracies across the scantypes
    '''
    dfs = []
    for code in DATA_CONFIG.CODES:
        df = calculate_performance(code, MAE)
        full_model_name = complete_name + CODE_TO_SCANTYPE[code]
        df.rename(index={0: full_model_name}, inplace=True)
        #display(HTML(df.to_html()))
        dfs.append(df)

    result = pd.concat(dfs)
    result.index.name = 'Model_Scantype'
    result = result.round(2)
    logging.info(result)

    # Save the model results in csv file
    result.to_csv(CSV_OUT_PATH, index=True)


def load_depth(fpath: str) -> Tuple[bytes, int, int, float, float]:
    """Take ZIP file and extract depth and metadata
    Args:
        fpath (str): File path to the ZIP
    Returns:
        depth_data (bytes): depthmap data
        width(int): depthmap width in pixel
        height(int): depthmap height in pixel
        depth_scale(float)
        max_confidence(float)
    """
    with zipfile.ZipFile(fpath) as z:
        with z.open('data') as f:
            # Example for a first_line: '180x135_0.001_7_0.57045287_-0.0057296_0.0022602521_0.82130724_-0.059177425_0.0024800065_0.030834956'
            first_line = f.readline().decode().strip()

            file_header = first_line.split("_")

            # header[0] example: 180x135
            width, height = file_header[0].split("x")
            width, height = int(width), int(height)
            depth_scale = float(file_header[1])
            max_confidence = float(file_header[2])

            depth_data = f.read()
    return depth_data, width, height, depth_scale, max_confidence


def parse_depth(tx, ty, data, depthScale):
    depth = data[(int(ty) * width + int(tx)) * 3 + 0] << 8
    depth += data[(int(ty) * width + int(tx)) * 3 + 1]
    depth *= depthScale
    return depth


def prepare_depthmap(data: bytes, width: int, height: int, depth_scale: float) -> np.array:
    """Convert bytes array into np.array"""
    output = np.zeros((width, height, 1))
    for cx in range(width):
        for cy in range(height):
            # depth data scaled to be visible
            output[cx][height - cy - 1] = parse_depth(cx, cy, data, depth_scale, width)
    arr = np.array(output, dtype='float32')
    return arr.reshape(width, height)


def preprocess(depthmap):
    #logging.info(depthmap.dtype)
    depthmap = preprocess_depthmap(depthmap)
    #depthmap = depthmap/depthmap.max()
    depthmap = depthmap / 7.5
    depthmap = resize(depthmap, (image_target_height, image_target_width))
    depthmap = depthmap.reshape((depthmap.shape[0], depthmap.shape[1], 1))
    #depthmap = depthmap[None, :]
    return depthmap


def setWidth(value):
    global width
    width = value


def setHeight(value):
    global height
    height = value


def download_model(workspace, experiment_name, run_id, input_location, output_location):
    '''
    Download the pretrained model
    Input:
         workspace: workspace to access the experiment
         experiment_name: Name of the experiment in which model is saved
         run_id: Run Id of the experiment in which model is pre-trained
         input_location: Input location in a RUN Id
         output_location: Location for saving the model
    '''
    experiment = Experiment(workspace=workspace, name=experiment_name)
    #Download the model on which evaluation need to be done
    run = Run(experiment, run_id=run_id)
    #run.get_details()

    if input_location.endswith(".h5"):
        run.download_file(input_location, output_location)
    elif input_location.endswith(".ckpt"):
        run.download_files(prefix=input_location, output_directory=output_location)
    else:
        raise NameError(f"{input_location}'s path extension not supported")
    logging.info("Successfully downloaded model")
