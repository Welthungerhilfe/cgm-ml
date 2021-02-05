import argparse
import copy
import os
import pickle
import random
import time
from importlib import import_module
from pathlib import Path

import glob2 as glob
import numpy as np
import pandas as pd
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
from tensorflow.keras.models import Sequential, load_model
from tensorflow.python import keras

from utils import get_run_ids, download_model

import utils
from constants import DATA_DIR_ONLINE_RUN, DEFAULT_CONFIG, REPO_DIR
from utils import (AGE_IDX, COLUMN_NAME_AGE, COLUMN_NAME_GOODBAD,
                   COLUMN_NAME_SEX, GOODBAD_IDX, GOODBAD_DICT, SEX_IDX,
                   calculate_performance, calculate_performance_age,
                   calculate_performance_goodbad, calculate_performance_sex,
                   download_dataset, draw_age_scatterplot,
                   draw_uncertainty_goodbad_plot, get_dataset_path,
                   get_model_path, draw_uncertainty_scatterplot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_config_module", default=DEFAULT_CONFIG, help="Configuration file")
    args = parser.parse_args()
    print(f"Using qa_config {args.qa_config_module}")
    qa_config = import_module(args.qa_config_module)
else:
    print(f"Using qa_config {DEFAULT_CONFIG}")
    qa_config = import_module(DEFAULT_CONFIG)

MODEL_CONFIG = qa_config.MODEL_CONFIG
EVAL_CONFIG = qa_config.EVAL_CONFIG
DATA_CONFIG = qa_config.DATA_CONFIG
RESULT_CONFIG = qa_config.RESULT_CONFIG
FILTER_CONFIG = qa_config.FILTER_CONFIG if getattr(qa_config, 'FILTER_CONFIG', False) else None


#RUN_IDS = MODEL_CONFIG.RUN_IDS
#print(f"Using runs: {RUN_IDS}")

# Function for loading and processing depthmaps.
def tf_load_pickle(path, max_value):
    """Utility to load the depthmap pickle file"""
    def py_load_pickle(path, max_value):
        if FILTER_CONFIG is not None:
            depthmap, targets, image = pickle.load(open(path.numpy(), "rb"))  # for filter (Contains RGBs)
        else:
            depthmap, targets = pickle.load(open(path.numpy(), "rb"))
        depthmap = utils.preprocess_depthmap(depthmap)
        depthmap = depthmap / max_value
        depthmap = tf.image.resize(depthmap, (DATA_CONFIG.IMAGE_TARGET_HEIGHT, DATA_CONFIG.IMAGE_TARGET_WIDTH))
        targets = utils.preprocess_targets(targets, DATA_CONFIG.TARGET_INDEXES)
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
    depthmap.set_shape((DATA_CONFIG.IMAGE_TARGET_HEIGHT, DATA_CONFIG.IMAGE_TARGET_WIDTH, 1))
    targets.set_shape((len(DATA_CONFIG.TARGET_INDEXES,)))
    return path, depthmap, targets


def prepare_sample_dataset(df_sample, dataset_path):
    df_sample['artifact_path'] = df_sample.apply(
        lambda x: f"{dataset_path}/{x['qrcode']}/{x['scantype']}/{x['artifact']}", axis=1)
    paths_evaluation = list(df_sample['artifact_path'])
    dataset_sample = tf.data.Dataset.from_tensor_slices(paths_evaluation)
    dataset_sample = dataset_sample.map(lambda path: tf_load_pickle(path, DATA_CONFIG.NORMALIZATION_VALUE))
    dataset_sample = dataset_sample.map(lambda _path, depthmap, targets: (depthmap, targets))
    dataset_sample = dataset_sample.cache()
    dataset_sample = dataset_sample.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset_sample


def predict_uncertainty(X: np.array, model: tf.keras.Model) -> float:
    """Predict standard deviation of multiple predictions with different dropouts

    Args:
        X: Sample image with shape (1, h, w, 1)
        model: keras model

    Returns:
        The standard deviation of multiple predictions
    """
    one_batch = np.repeat(X, RESULT_CONFIG.NUM_DROPOUT_PREDICTIONS, axis=0)
    predictions = model(one_batch, training=True)
    std = tf.math.reduce_std(predictions)
    return std


def change_dropout_strength(model: tf.keras.Model, dropout_strength: float) -> tf.keras.Model:
    """Duplicate a model while adjusting the dropout rate"""
    new_model = Sequential(name="new_model")
    for layer_ in model.layers:
        layer = copy.copy(layer_)
        if isinstance(layer, keras.layers.core.Dropout):
            # Set the dropout rate a ratio from range [0.0, 1.0]
            layer.rate = min(0.999, layer.rate * dropout_strength)
        new_model.add(layer)
    return new_model


def get_prediction_uncertainty(model_paths: list, dataset_evaluation: tf.data.Dataset) -> np.array:
    """Predict standard deviation of multiple predictions with different dropouts

    Args:
        model_path: Path of the trained model
        dataset_evaluation: dataset in which the evaluation need to performed

    Returns:
        predictions, array shape (N_SAMPLES, )
    """

    dataset = dataset_evaluation.batch(1)

    print("starting predicting uncertainty")

    # Go through all models and compute STD of predictions.
    start = time.time()
    std_list = []
    for model_path in model_paths:
        print(f"loading model from {model_path}")
        model = load_model(model_path, compile=False)
        std_list += [[model.predict(X)[0] for X, y in dataset.as_numpy_iterator()]]
    std_list = np.array(std_list)
    std_list = np.std(std_list, axis=0)
    std_list = std_list.reshape((-1))
    end = time.time()
    print(f"Total time for uncertainty prediction experiment: {end - start:.3} sec")

    return np.array(std_list)


def get_prediction(model_path: str, dataset_evaluation: tf.data.Dataset) -> np.array:
    """Perform the prediction on the dataset with the given model

    Args:
        model_path: Path of the trained model
        dataset_evaluation: dataset in which the evaluation need to performed
    Returns:
        predictions, array shape (N_SAMPLES, )
    """
    print(f"loading model from {model_path}")
    model = load_model(model_path, compile=False)

    dataset = dataset_evaluation.batch(DATA_CONFIG.BATCH_SIZE)

    print("starting predicting")
    start = time.time()
    predictions = model.predict(dataset, batch_size=DATA_CONFIG.BATCH_SIZE)
    end = time.time()
    print(f"Total time for prediction experiment: {end - start:.3} sec")

    prediction_list = np.squeeze(predictions)
    return prediction_list


if __name__ == "__main__":

    # Make experiment reproducible
    tf.random.set_seed(EVAL_CONFIG.SPLIT_SEED)
    random.seed(EVAL_CONFIG.SPLIT_SEED)

    # Get the current run.
    run = Run.get_context()
    print(f"Using run {run}")

    OUTPUT_CSV_PATH = str(REPO_DIR / 'data' / RESULT_CONFIG.SAVE_PATH) if run.id.startswith("OfflineRun") else RESULT_CONFIG.SAVE_PATH
    print(f"OUTPUT_CSV_PATH: {OUTPUT_CSV_PATH}")

    MODEL_BASE_DIR = REPO_DIR / 'data' if run.id.startswith("OfflineRun") else Path('.')
    MODEL_BASE_DIR = Path(".") # TODO remove this!
    
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
            print("Downloading...")
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

    print(f"MODEL_BASE_DIR: {MODEL_BASE_DIR}")

    # Get run ids. Either defined by the user or get all.
    run_ids = MODEL_CONFIG.RUN_IDS
    if run_ids == "all":
        run_ids = get_run_ids(ws=workspace, experiment_name=MODEL_CONFIG.EXPERIMENT_NAME)
    print(f"Using run ids: {run_ids}")

    # Download all models.
    for run_id in run_ids:
        print(f"Downloading run {run_id}")
        download_model(
            ws=workspace,
            experiment_name=MODEL_CONFIG.EXPERIMENT_NAME,
            run_id=run_id,
            input_location=os.path.join(MODEL_CONFIG.INPUT_LOCATION, MODEL_CONFIG.NAME),
            output_location=os.path.join(MODEL_BASE_DIR, run_id)
        )

    # TODO Why do I have to append the path?
    model_paths = [os.path.join(path, "outputs", "best_model.ckpt") for path in glob.glob(os.path.join(MODEL_BASE_DIR, "*")) if os.path.isdir(path) and path.split("/")[-1].startswith(MODEL_CONFIG.EXPERIMENT_NAME)]
    #model_paths = model_paths[0:2] # TODO remove this!
    print(f"Models paths ({len(model_paths)}):")
    print("\t" + "\n\t".join(model_paths))
    del MODEL_BASE_DIR



    # Get the QR-code paths.
    dataset_path = os.path.join(dataset_path, "scans")
    print("Dataset path:", dataset_path)
    # print(glob.glob(os.path.join(dataset_path, "*"))) # Debug
    print("Getting QR code paths...")
    qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
    #qrcode_paths = qrcode_paths[0:1] # TODO remove this!
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

    new_paths_evaluation = paths_evaluation

    if FILTER_CONFIG is not None and FILTER_CONFIG.IS_ENABLED:
        standing = load_model(FILTER_CONFIG.NAME)
        new_paths_evaluation = utils.filter_dataset(paths_evaluation, standing)

    print("Creating dataset for training.")
    paths = new_paths_evaluation
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset_norm = dataset.map(lambda path: tf_load_pickle(path, DATA_CONFIG.NORMALIZATION_VALUE))

    # filter goodbad==delete
    # TODO make this work again!
    #dataset_norm = dataset_norm.filter(lambda _path, _depthmap, targets: targets[2] != GOODBAD_DICT['delete'])  # TODO refactor: replace 2 with inferred goodbad target idx

    dataset_norm = dataset_norm.cache()
    dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
    tmp_dataset_evaluation = dataset_norm
    del dataset_norm
    print("Created dataset for training.")


    # Update new_paths_evaluation after filtering
    dataset_paths = tmp_dataset_evaluation.map(lambda path, _depthmap, _targets: path)
    list_paths = list(dataset_paths.as_numpy_iterator())
    new_paths_evaluation = [x.decode() for x in list_paths]

    dataset_evaluation = tmp_dataset_evaluation.map(lambda _path, depthmap, targets: (depthmap, targets))
    del tmp_dataset_evaluation

    # TODO make this work. Either have 16 predictions or somehow merge.
    # TODO I guess compute mean and std
    print("Predicting...")
    prediction_list_one = []
    for model_index, model_path in enumerate(model_paths):
        print(f"Model {model_index + 1}/{len(model_paths)}")
        prediction_list_one += [get_prediction(model_path, dataset_evaluation)]
        print("Prediction made by model on the depthmaps...")
    prediction_list_one = np.array(prediction_list_one)
    prediction_list_mean = np.mean(prediction_list_one, axis=0) 
    print(prediction_list_mean)
    del prediction_list_one


    # Get column data. Includes the targets from the pickle files.
    qrcode_list, scantype_list, artifact_list, prediction_list_mean, target_list = utils.get_column_list(
        new_paths_evaluation, prediction_list_mean, DATA_CONFIG, FILTER_CONFIG)

    # Create the data frame.
    df = pd.DataFrame({
        'qrcode': qrcode_list,
        'artifact': artifact_list,
        'scantype': scantype_list,
        'GT': [el for el in target_list],
        'predicted': prediction_list_mean,
    }, columns=RESULT_CONFIG.COLUMNS)
    print("df.shape:", df.shape)

    # Set some types.
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

    # We use multiple run ids.
    RUN_ID = "ALL RUNS"

    csv_file = f"{OUTPUT_CSV_PATH}/{RUN_ID}.csv"
    print(f"Calculate and save the results to {csv_file}")
    utils.calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_file,
                                     DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance)
    if 'AGE_BUCKETS' in RESULT_CONFIG.keys():
        csv_file = f"{OUTPUT_CSV_PATH}/age_evaluation_{RUN_ID}.csv"
        print(f"Calculate and save age results to {csv_file}")
        utils.calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_file,
                                         DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance_age)

        png_file = f"{OUTPUT_CSV_PATH}/age_evaluation_scatter_{RUN_ID}.png"
        print(f"Calculate and save scatterplot results to {png_file}")
        draw_age_scatterplot(df, png_file)

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

    # Now evaluate.

    # TODO Make this work again.
    #assert GOODBAD_IDX in DATA_CONFIG.TARGET_INDEXES
    #assert COLUMN_NAME_GOODBAD in df

    # Sample one artifact per scan (qrcode, scantype combination)
    df_sample = df.groupby(['qrcode', 'scantype']).apply(lambda x: x.sample(1))

    # Prepare uncertainty prediction on these artifacts
    dataset_sample = prepare_sample_dataset(df_sample, dataset_path)

    # Predict uncertainty
    uncertainties = get_prediction_uncertainty(model_paths, dataset_sample)
    assert len(df_sample) == len(uncertainties)
    df_sample['uncertainties'] = uncertainties

    # TODO make this work again.
    #png_file = f"{OUTPUT_CSV_PATH}/uncertainty_distribution_dropoutstrength{RESULT_CONFIG.DROPOUT_STRENGTH}_{RUN_ID}.png"
    #draw_uncertainty_goodbad_plot(df_sample, png_file)

    # TODO make this work again.
    #df_sample_100 = df_sample.iloc[df_sample.index.get_level_values('scantype') == '100']
    #png_file = f"{OUTPUT_CSV_PATH}/uncertainty_code100_distribution_dropoutstrength{RESULT_CONFIG.DROPOUT_STRENGTH}_{RUN_ID}.png"
    #draw_uncertainty_goodbad_plot(df_sample_100, png_file)

    png_file = f"{OUTPUT_CSV_PATH}/uncertainty_scatter_distribution_{RUN_ID}.png"
    draw_uncertainty_scatterplot(df_sample, png_file)
    print(png_file)

    # Done.
    run.complete()