import argparse
import os
import pickle
import random
import time
from pathlib import Path
from importlib import import_module

import glob2 as glob
import numpy as np
import pandas as pd
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
from tensorflow.keras.models import load_model

import utils
from constants import DATA_DIR_ONLINE_RUN, REPO_DIR, DEFAULT_CONFIG
from utils import (AGE_IDX, COLUMN_NAME_AGE, COLUMN_NAME_GOODBAD,
                   COLUMN_NAME_SEX, GOODBAD_IDX, SEX_IDX,
                   calculate_performance, calculate_performance_age,
                   calculate_performance_goodbad, calculate_performance_sex,
                   download_dataset, draw_age_scatterplot, get_dataset_path,
                   get_model_path)

parser = argparse.ArgumentParser()
parser.add_argument("--qa_config_module", default=DEFAULT_CONFIG, help="Configuration file")
args = parser.parse_args()

qa_config = import_module(args.qa_config_module)
MODEL_CONFIG = qa_config.MODEL_CONFIG
EVAL_CONFIG = qa_config.EVAL_CONFIG
DATA_CONFIG = qa_config.DATA_CONFIG
RESULT_CONFIG = qa_config.RESULT_CONFIG

RUN_ID = MODEL_CONFIG.RUN_ID


# Function for loading and processing depthmaps.
def tf_load_pickle(path, max_value):
    """Utility to load the depthmap pickle file"""
    def py_load_pickle(path, max_value):
        depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = utils.preprocess_depthmap(depthmap)
        depthmap = depthmap / max_value
        depthmap = tf.image.resize(depthmap, (DATA_CONFIG.IMAGE_TARGET_HEIGHT, DATA_CONFIG.IMAGE_TARGET_WIDTH))
        targets = utils.preprocess_targets(targets, DATA_CONFIG.TARGET_INDEXES)
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
    depthmap.set_shape((DATA_CONFIG.IMAGE_TARGET_HEIGHT, DATA_CONFIG.IMAGE_TARGET_WIDTH, 1))
    targets.set_shape((len(DATA_CONFIG.TARGET_INDEXES,)))
    return depthmap, targets


def get_prediction(model_path, dataset_evaluation):
    """Perform the prediction on the dataset with the given model

    Args:
        model_path: Path of the trained model
        dataset_evaluation: dataset in which Evaluation need to performed

    Returns:
        prediction_list
    """
    model = load_model(model_path, compile=False)

    dataset = dataset_evaluation.batch(DATA_CONFIG.BATCH_SIZE)

    print("starting predicting")
    start = time.time()
    predictions = model.predict(dataset, batch_size=DATA_CONFIG.BATCH_SIZE)
    end = time.time()
    print("Total time for prediction experiment: {} sec".format(end - start))

    prediction_list = np.squeeze(predictions)
    return prediction_list


if __name__ == "__main__":

    # Make experiment reproducible
    tf.random.set_seed(EVAL_CONFIG.SPLIT_SEED)
    random.seed(EVAL_CONFIG.SPLIT_SEED)

    # Get the current run.
    run = Run.get_context()

    OUTPUT_CSV_PATH = str(REPO_DIR / RESULT_CONFIG.SAVE_PATH) if run.id.startswith("OfflineRun") else RESULT_CONFIG.SAVE_PATH
    MODEL_BASE_DIR = REPO_DIR / 'data' / MODEL_CONFIG.RUN_ID if run.id.startswith("OfflineRun") else Path('.')

    # Offline run. Download the sample dataset and run locally. Still push results to Azure.
    if run.id.startswith("OfflineRun"):
        print("Running in offline mode...")

        # Access workspace.
        print("Accessing workspace...")
        workspace = Workspace.from_config()
        experiment = Experiment(workspace, EVAL_CONFIG.EXPERIMENT_NAME)
        run = experiment.start_logging(outputs=None, snapshot_directory=None)

        # Get dataset.
        print("Accessing dataset...")
        dataset_name = DATA_CONFIG.NAME
        dataset_path = str(REPO_DIR / "data" / dataset_name)
        if not os.path.exists(dataset_path):
            dataset = workspace.datasets[dataset_name]
            dataset.download(target_path=dataset_path, overwrite=False)

    # Online run. Use dataset provided by training notebook.
    else:
        print("Running in online mode...")
        experiment = run.experiment
        workspace = experiment.workspace
        dataset_name = DATA_CONFIG.NAME

        # Download
        dataset_path = get_dataset_path(DATA_DIR_ONLINE_RUN, dataset_name)
        download_dataset(workspace, dataset_name, dataset_path)

    # Get the QR-code paths.
    dataset_path = os.path.join(dataset_path, "scans")
    print("Dataset path:", dataset_path)
    #print(glob.glob(os.path.join(dataset_path, "*"))) # Debug
    print("Getting QR code paths...")
    qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
    print("QR code paths: ", len(qrcode_paths))
    assert len(qrcode_paths) != 0

    if EVAL_CONFIG.DEBUG_RUN and len(qrcode_paths) > EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN:
        qrcode_paths = qrcode_paths[:EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN]
        print("Executing on {} qrcodes for FAST RUN".format(EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN))

    print("Paths for evaluation:")
    print("\t" + "\n\t".join(qrcode_paths))

    print(len(qrcode_paths))

    # Get the pointclouds.
    print("Getting Depthmap paths...")
    paths_evaluation = utils.get_depthmap_files(qrcode_paths)
    del qrcode_paths

    print("Using {} artifact files for evaluation.".format(len(paths_evaluation)))

    print("Creating dataset for training.")
    paths = paths_evaluation
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset_norm = dataset.map(lambda path: tf_load_pickle(path, DATA_CONFIG.NORMALIZATION_VALUE))
    dataset_norm = dataset_norm.cache()
    dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
    dataset_evaluation = dataset_norm
    del dataset_norm
    print("Created dataset for training.")

    model_path = MODEL_BASE_DIR / get_model_path(MODEL_CONFIG)
    prediction_list_one = get_prediction(model_path, dataset_evaluation)

    print("Prediction made by model on the depthmaps...")
    print(prediction_list_one)

    qrcode_list, scantype_list, artifact_list, prediction_list, target_list = utils.get_column_list(
        paths_evaluation, prediction_list_one, DATA_CONFIG)

    df = pd.DataFrame({
        'qrcode': qrcode_list,
        'artifact': artifact_list,
        'scantype': scantype_list,
        'GT': [el[0] for el in target_list],
        'predicted': prediction_list
    }, columns=RESULT_CONFIG.COLUMNS)

    df['GT'] = df['GT'].astype('float64')
    df['predicted'] = df['predicted'].astype('float64')

    if 'AGE_BUCKETS' in RESULT_CONFIG.keys():
        idx = DATA_CONFIG.TARGET_INDEXES.index(AGE_IDX)
        df[COLUMN_NAME_AGE] = [el[idx] for el in target_list]
    if SEX_IDX in DATA_CONFIG.TARGET_INDEXES:
        idx = DATA_CONFIG.TARGET_INDEXES.index(SEX_IDX)
        df[COLUMN_NAME_SEX] = [el[idx] for el in target_list]
    if GOODBAD_IDX in DATA_CONFIG.TARGET_INDEXES:
        idx = DATA_CONFIG.TARGET_INDEXES.index(GOODBAD_IDX)
        df[COLUMN_NAME_GOODBAD] = [el[idx] for el in target_list]

    df_grouped = df.groupby(['qrcode', 'scantype']).mean()
    print("Mean Avg Error: ", df_grouped)

    df_grouped['error'] = df_grouped.apply(utils.avgerror, axis=1)

    csv_file = f"{OUTPUT_CSV_PATH}/{RUN_ID}.csv"
    print(f"Calculate and save the results to {csv_file}")
    utils.calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_file,
                                     DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance)

    if 'AGE_BUCKETS' in RESULT_CONFIG.keys():
        csv_file = f"{OUTPUT_CSV_PATH}/age_evaluation_{RUN_ID}.csv"
        print(f"Calculate and save age results to {csv_file}")
        utils.calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_file,
                                         DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance_age)

        csv_file = f"{OUTPUT_CSV_PATH}/age_evaluation_scatter_{RUN_ID}.png"
        print(f"Calculate and save scatterplot results to {csv_file}")
        draw_age_scatterplot(df, csv_file)

    if SEX_IDX in DATA_CONFIG.TARGET_INDEXES:
        csv_file = f"{OUTPUT_CSV_PATH}/sex_evaluation_{RUN_ID}.csv"
        print(f"Calculate and save sex results to {csv_file}")
        utils.calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_file,
                                         DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance_sex)
    if GOODBAD_IDX in DATA_CONFIG.TARGET_INDEXES:
        csv_file = f"{OUTPUT_CSV_PATH}/goodbad_evaluation_{RUN_ID}.csv"
        print(f"Calculate performance on bad/good scans and save results to {csv_file}")
        utils.calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_file,
                                         DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance_goodbad)

    # Done.
    run.complete()
