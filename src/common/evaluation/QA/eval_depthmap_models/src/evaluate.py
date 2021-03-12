import argparse
import copy
import os
import logging
import logging.config
import pickle
import random
import time
from importlib import import_module
from pathlib import Path
from typing import List
import shutil

import glob2 as glob
import numpy as np
import pandas as pd
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
from tensorflow.keras.models import Sequential, load_model
from tensorflow.python import keras

import utils
from constants import DATA_DIR_ONLINE_RUN, DEFAULT_CONFIG, REPO_DIR
from utils import (AGE_IDX, COLUMN_NAME_AGE, COLUMN_NAME_GOODBAD, HEIGHT_IDX, WEIGHT_IDX,
                   COLUMN_NAME_SEX, GOODBAD_IDX, GOODBAD_DICT, SEX_IDX, avgerror,
                   calculate_performance, calculate_performance_age,
                   calculate_performance_goodbad, calculate_performance_sex,
                   download_dataset, draw_age_scatterplot, draw_stunting_diagnosis,
                   draw_uncertainty_goodbad_plot, extract_scantype, extract_qrcode,
                   get_dataset_path, draw_wasting_diagnosis,
                   get_model_path, draw_uncertainty_scatterplot)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_config_module", default=DEFAULT_CONFIG, help="Configuration file")
    args = parser.parse_args()
    qa_config = import_module(args.qa_config_module)
else:
    qa_config = import_module(DEFAULT_CONFIG)


MODEL_CONFIG = qa_config.MODEL_CONFIG
EVAL_CONFIG = qa_config.EVAL_CONFIG
DATA_CONFIG = qa_config.DATA_CONFIG
RESULT_CONFIG = qa_config.RESULT_CONFIG
FILTER_CONFIG = qa_config.FILTER_CONFIG if getattr(qa_config, 'FILTER_CONFIG', False) else None


RUN_ID = MODEL_CONFIG.RUN_ID

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


def get_prediction_uncertainty(model_path: str, dataset_evaluation: tf.data.Dataset) -> np.array:
    """Predict standard deviation of multiple predictions with different dropouts

    Args:
        model_path: Path of the trained model
        dataset_evaluation: dataset in which the evaluation need to performed

    Returns:
        predictions, array shape (N_SAMPLES, )
    """
    logging.info("loading model from %s", model_path)
    model = load_model(model_path, compile=False)
    model = change_dropout_strength(model, RESULT_CONFIG.DROPOUT_STRENGTH)

    dataset = dataset_evaluation.batch(1)

    logging.info("starting predicting uncertainty")
    start = time.time()
    std_list = [predict_uncertainty(X, model) for X, y in dataset.as_numpy_iterator()]
    end = time.time()
    logging.info("Total time for uncertainty prediction experiment: %d sec", end - start)

    return np.array(std_list)


def get_prediction_multiartifact(model_path: str, samples_paths: List[List[str]]) -> List[List[str]]:
    """

    Args:
        model_path: [description]
        samples_paths: [description]

    Returns:
        List with tuples: ('artifacts', 'predicted', 'GT')
    """

    logging.info("loading model from %s", model_path)
    model = load_model(model_path, compile=False)

    predictions = []
    for sample_paths in samples_paths:
        depthmap, targets = create_multiartifact_sample(sample_paths,
                                                        DATA_CONFIG.NORMALIZATION_VALUE,
                                                        DATA_CONFIG.IMAGE_TARGET_HEIGHT,
                                                        DATA_CONFIG.IMAGE_TARGET_WIDTH,
                                                        DATA_CONFIG.TARGET_INDEXES)
        depthmaps = tf.stack([depthmap])
        pred = model.predict(depthmaps)
        predictions.append([sample_paths[0], float(np.squeeze(pred)), targets[0]])
    return predictions


def get_prediction(model_path: str, dataset_evaluation: tf.data.Dataset) -> np.array:
    """Perform the prediction on the dataset with the given model

    Args:
        model_path: Path of the trained model
        dataset_evaluation: dataset in which the evaluation need to performed
    Returns:
        predictions, array shape (N_SAMPLES, )
    """
    logging.info("loading model from %s", model_path)
    model = load_model(model_path, compile=False)

    dataset = dataset_evaluation.batch(DATA_CONFIG.BATCH_SIZE)

    logging.info("starting predicting")
    start = time.time()
    predictions = model.predict(dataset, batch_size=DATA_CONFIG.BATCH_SIZE)
    end = time.time()
    logging.info("Total time for uncertainty prediction experiment: %d sec", end - start)

    prediction_list = np.squeeze(predictions)
    return prediction_list


if __name__ == "__main__":

    # Make experiment reproducible
    tf.random.set_seed(EVAL_CONFIG.SPLIT_SEED)
    random.seed(EVAL_CONFIG.SPLIT_SEED)

    # Get the current run.
    run = Run.get_context()

    if run.id.startswith("OfflineRun"):
        utils_dir_path = REPO_DIR / "src/common/model_utils"
        utils_paths = glob.glob(os.path.join(utils_dir_path, "*.py"))
        temp_model_utils_dir = Path(__file__).parent / "tmp_model_utils"
        # Remove old temp_path
        if os.path.exists(temp_model_utils_dir):
            shutil.rmtree(temp_model_utils_dir)
        # Copy
        os.mkdir(temp_model_utils_dir)
        os.system(f'touch {temp_model_utils_dir}/__init__.py')
        for p in utils_paths:
            shutil.copy(p, temp_model_utils_dir)

    from tmp_model_utils.preprocessing import preprocess_depthmap, preprocess_targets, create_samples  # noqa: E402, F401
    from preprocessing_multiartifact import create_multiartifact_sample  # noqa: E402, F401

    OUTPUT_CSV_PATH = str(REPO_DIR / 'data'
                          / RESULT_CONFIG.SAVE_PATH) if run.id.startswith("OfflineRun") else RESULT_CONFIG.SAVE_PATH
    MODEL_BASE_DIR = REPO_DIR / 'data' / MODEL_CONFIG.RUN_ID if run.id.startswith("OfflineRun") else Path('.')

    # Offline run. Download the sample dataset and run locally. Still push results to Azure.
    if run.id.startswith("OfflineRun"):
        logging.info("Running in offline mode...")

        # Access workspace.
        logging.info("Accessing workspace...")
        workspace = Workspace.from_config()
        experiment = Experiment(workspace, EVAL_CONFIG.EXPERIMENT_NAME)
        run = experiment.start_logging(outputs=None, snapshot_directory=None)

        # Get dataset.
        logging.info("Accessing dataset...")
        dataset_name = DATA_CONFIG.NAME
        dataset_path = str(REPO_DIR / "data" / dataset_name)
        if not os.path.exists(dataset_path):
            dataset = workspace.datasets[dataset_name]
            dataset.download(target_path=dataset_path, overwrite=False)

    # Online run. Use dataset provided by training notebook.
    else:
        logging.info("Running in online mode...")
        experiment = run.experiment
        workspace = experiment.workspace
        dataset_name = DATA_CONFIG.NAME

        # Download
        dataset_path = get_dataset_path(DATA_DIR_ONLINE_RUN, dataset_name)
        download_dataset(workspace, dataset_name, dataset_path)

    # Get the QR-code paths.
    dataset_path = os.path.join(dataset_path, "scans")
    logging.info('Dataset path: %s', dataset_path)
    #logging.info(glob.glob(os.path.join(dataset_path, "*"))) # Debug
    logging.info('Getting QR-code paths...')
    qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
    logging.info('qrcode_paths: %d', len(qrcode_paths))
    assert len(qrcode_paths) != 0

    if EVAL_CONFIG.DEBUG_RUN and len(qrcode_paths) > EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN:
        qrcode_paths = qrcode_paths[:EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN]
        logging.info("Executing on %d qrcodes for FAST RUN", EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN)

    logging.info('Paths for evaluation: \n\t' + '\n\t'.join(qrcode_paths))
    logging.info(len(qrcode_paths))

    model_path = MODEL_BASE_DIR / get_model_path(MODEL_CONFIG)

    # Is this a multiartifact model?
    if getattr(DATA_CONFIG, "N_ARTIFACTS", 1) > 1:
        samples_paths = create_samples(qrcode_paths, DATA_CONFIG)
        predictions = get_prediction_multiartifact(model_path, samples_paths)

        df = pd.DataFrame(predictions, columns=['artifacts', 'predicted', 'GT'])
        df['scantype'] = df.apply(extract_scantype, axis=1)
        df['qrcode'] = df.apply(extract_qrcode, axis=1)
        MAE = df.groupby(['qrcode', 'scantype']).mean()
        MAE['error'] = MAE.apply(avgerror, axis=1)

    else:  # Single-artifact

        # Get the pointclouds.
        logging.info("Getting Depthmap paths...")
        paths_evaluation = utils.get_depthmap_files(qrcode_paths)
        del qrcode_paths

        logging.info("Using %d artifact files for evaluation.", len(paths_evaluation))

        new_paths_evaluation = paths_evaluation

        if FILTER_CONFIG is not None and FILTER_CONFIG.IS_ENABLED:
            standing = load_model(FILTER_CONFIG.NAME)
            new_paths_evaluation = utils.filter_dataset(paths_evaluation, standing)

        logging.info("Creating dataset for training.")
        paths = new_paths_evaluation
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        dataset_norm = dataset.map(lambda path: tf_load_pickle(path, DATA_CONFIG.NORMALIZATION_VALUE))

        # filter goodbad==delete
        if GOODBAD_IDX in DATA_CONFIG.TARGET_INDEXES:
            goodbad_index = DATA_CONFIG.TARGET_INDEXES.index(GOODBAD_IDX)
            dataset_norm = dataset_norm.filter(
                lambda _path, _depthmap, targets: targets[goodbad_index] != GOODBAD_DICT['delete'])

        dataset_norm = dataset_norm.cache()
        dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
        tmp_dataset_evaluation = dataset_norm
        del dataset_norm
        logging.info("Created dataset for training.")

        # Update new_paths_evaluation after filtering
        dataset_paths = tmp_dataset_evaluation.map(lambda path, _depthmap, _targets: path)
        list_paths = list(dataset_paths.as_numpy_iterator())
        new_paths_evaluation = [x.decode() for x in list_paths]

        dataset_evaluation = tmp_dataset_evaluation.map(lambda _path, depthmap, targets: (depthmap, targets))
        del tmp_dataset_evaluation

        prediction_list_one = get_prediction(model_path, dataset_evaluation)
        logging.info("Prediction made by model on the depthmaps...")
        logging.info(prediction_list_one)

        qrcode_list, scantype_list, artifact_list, prediction_list, target_list = utils.get_column_list(
            new_paths_evaluation, prediction_list_one, DATA_CONFIG, FILTER_CONFIG)

        df = pd.DataFrame({
            'qrcode': qrcode_list,
            'artifact': artifact_list,
            'scantype': scantype_list,
            'GT': [el[0] for el in target_list],
            'predicted': prediction_list
        }, columns=RESULT_CONFIG.COLUMNS)
        logging.info("df.shape: %s", df.shape)

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
    logging.info("Mean Avg Error: %d", df_grouped)

    df_grouped['error'] = df_grouped.apply(utils.avgerror, axis=1)

    csv_file = f"{OUTPUT_CSV_PATH}/{RUN_ID}.csv"
    logging.info("Calculate and save the results to %s", csv_file)
    utils.calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_file,
                                     DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance)
    if 'AGE_BUCKETS' in RESULT_CONFIG.keys():
        csv_file = f"{OUTPUT_CSV_PATH}/age_evaluation_{RUN_ID}.csv"
        logging.info("Calculate and save age results to %s", csv_file)
        utils.calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_file,
                                         DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance_age)
        png_file = f"{OUTPUT_CSV_PATH}/age_evaluation_scatter_{RUN_ID}.png"
        logging.info("Calculate and save scatterplot results to %s", png_file)
        draw_age_scatterplot(df, png_file)

    if HEIGHT_IDX in DATA_CONFIG.TARGET_INDEXES and 'AGE_BUCKETS' in RESULT_CONFIG.keys():
        png_file = f"{OUTPUT_CSV_PATH}/stunting_diagnosis_{RUN_ID}.png"
        logging.info("Calculate zscores and save confusion matrix results to %s", png_file)
        start = time.time()
        draw_stunting_diagnosis(df, png_file)
        end = time.time()
        logging.info("Total time for Calculate zscores and save confusion matrix: %d", end - start)

    if WEIGHT_IDX in DATA_CONFIG.TARGET_INDEXES and 'AGE_BUCKETS' in RESULT_CONFIG.keys():
        png_file = f"{OUTPUT_CSV_PATH}/wasting_diagnosis_{RUN_ID}.png"
        logging.info("Calculate and save wasting confusion matrix results to %s", png_file)
        start = time.time()
        draw_wasting_diagnosis(df, png_file)
        end = time.time()
        logging.info("Total time for Calculate zscores and save wasting confusion matrix: %d", end - start)

    if SEX_IDX in DATA_CONFIG.TARGET_INDEXES:
        csv_file = f"{OUTPUT_CSV_PATH}/sex_evaluation_{RUN_ID}.csv"
        logging.info("Calculate and save sex results to %s", csv_file)
        utils.calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_file,
                                         DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance_sex)
    if GOODBAD_IDX in DATA_CONFIG.TARGET_INDEXES:
        csv_file = f"{OUTPUT_CSV_PATH}/goodbad_evaluation_{RUN_ID}.csv"
        logging.info("Calculate performance on bad/good scans and save results to %s", csv_file)
        utils.calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_file,
                                         DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance_goodbad)

    if RESULT_CONFIG.USE_UNCERTAINTY:
        assert GOODBAD_IDX in DATA_CONFIG.TARGET_INDEXES
        assert COLUMN_NAME_GOODBAD in df

        # Sample one artifact per scan (qrcode, scantype combination)
        df_sample = df.groupby(['qrcode', 'scantype']).apply(lambda x: x.sample(1))

        # Prepare uncertainty prediction on these artifacts
        dataset_sample = prepare_sample_dataset(df_sample, dataset_path)

        # Predict uncertainty
        uncertainties = get_prediction_uncertainty(model_path, dataset_sample)
        assert len(df_sample) == len(uncertainties)
        df_sample['uncertainties'] = uncertainties

        png_file = f"{OUTPUT_CSV_PATH}/uncertainty_distribution_dropoutstrength{RESULT_CONFIG.DROPOUT_STRENGTH}_{RUN_ID}.png"
        draw_uncertainty_goodbad_plot(df_sample, png_file)

        df_sample_100 = df_sample.iloc[df_sample.index.get_level_values('scantype') == '100']
        png_file = f"{OUTPUT_CSV_PATH}/uncertainty_code100_distribution_dropoutstrength{RESULT_CONFIG.DROPOUT_STRENGTH}_{RUN_ID}.png"
        draw_uncertainty_goodbad_plot(df_sample_100, png_file)

        png_file = f"{OUTPUT_CSV_PATH}/uncertainty_scatter_distribution_{RUN_ID}.png"
        draw_uncertainty_scatterplot(df_sample, png_file)

        # Filter for scans with high certainty and calculate their accuracy/results
        df_sample['error'] = df_sample.apply(utils.avgerror, axis=1).abs()
        df_sample_better_threshold = df_sample[df_sample['uncertainties'] < RESULT_CONFIG.UNCERTAINTY_THRESHOLD_IN_CM]
        csv_file = f"{OUTPUT_CSV_PATH}/uncertainty_smaller_than_{RESULT_CONFIG.UNCERTAINTY_THRESHOLD_IN_CM}cm_{RUN_ID}.csv"
        logging.info("Uncertainty: For more certain than %d cm, calculate and save the results to %s", RESULT_CONFIG.UNCERTAINTY_THRESHOLD_IN_CM, csv_file)
        utils.calculate_and_save_results(df_sample_better_threshold, EVAL_CONFIG.NAME, csv_file,
                                         DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance)

    # Done.
    run.complete()
