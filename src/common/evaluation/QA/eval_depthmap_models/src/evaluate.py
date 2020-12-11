import argparse
from importlib import import_module
import os
import random
import pickle
import glob2 as glob
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
import utils
from utils import download_dataset, get_dataset_path

from constants import REPO_DIR, DATA_DIR_ONLINE_RUN

parser = argparse.ArgumentParser()
parser.add_argument("--qa_config_module", default="qa_config_height", help="Configuration file")
args = parser.parse_args()

qa_config = import_module(args.qa_config_module)
MODEL_CONFIG = qa_config.MODEL_CONFIG
EVAL_CONFIG = qa_config.EVAL_CONFIG
DATA_CONFIG = qa_config.DATA_CONFIG
RESULT_CONFIG = qa_config.RESULT_CONFIG


# Function for loading and processing depthmaps.
def tf_load_pickle(path, max_value):
    '''
    Utility to load the depthmap pickle file
    '''
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


def get_prediction(MODEL_PATH, dataset_evaluation):
    '''
    Perform the prediction on the dataset with the given model
    Input:
        MODEL_PATH : Path of the trained model
        dataset_evaluation : dataset in which Evaluation
        need to performed
    '''
    model = load_model(MODEL_PATH, compile=False)

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

    # Offline run. Download the sample dataset and run locally. Still push results to Azure.
    if(run.id.startswith("OfflineRun")):
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

    # Get the prediction
    if MODEL_CONFIG.NAME.endswith(".h5"):
        model_path = MODEL_CONFIG.NAME
    elif MODEL_CONFIG.NAME.endswith(".ckpt"):
        model_path = f"{MODEL_CONFIG.INPUT_LOCATION}/{MODEL_CONFIG.NAME}"
    else:
        raise NameError(f"{MODEL_CONFIG.NAME}'s path extension not supported")
    prediction_list_one = get_prediction(model_path, dataset_evaluation)

    print("Prediction made by model on the depthmaps...")
    print(prediction_list_one)

    qrcode_list, scantype_list, artifact_list, prediction_list, target_list = utils.get_column_list(
        paths_evaluation, prediction_list_one, DATA_CONFIG)

    df = pd.DataFrame({
        'qrcode': qrcode_list,
        'artifact': artifact_list,
        'scantype': scantype_list,
        'GT': target_list,
        'predicted': prediction_list
    }, columns=RESULT_CONFIG.COLUMNS)

    df['GT'] = df['GT'].astype('float64')
    df['predicted'] = df['predicted'].astype('float64')

    MAE = df.groupby(['qrcode', 'scantype']).mean()
    print("Mean Avg Error: ", MAE)

    MAE['error'] = MAE.apply(utils.avgerror, axis=1)

    print("Saving the results")
    utils.calculate_and_save_results(MAE, EVAL_CONFIG.NAME, RESULT_CONFIG.SAVE_PATH,
                                     DATA_CONFIG, RESULT_CONFIG, MODEL_CONFIG.RUN_ID)

    # Done.
    run.complete()
