import logging
import logging.config
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
import time
from typing import Callable, List, Tuple

import glob2 as glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from azureml.core import Experiment, Run, Workspace
from bunch import Bunch
from scipy.stats.stats import pearsonr
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from cgmzscore import Calculator  # noqa: E402

from .constants_eval import (  # noqa: E402
    CODE_TO_SCANTYPE, COLUMN_NAME_AGE, COLUMN_NAME_GOODBAD, COLUMN_NAME_SEX, DAYS_IN_YEAR,
    GOODBAD_IDX, GOODBAD_DICT, SEX_IDX, SEX_DICT, AGE_IDX, HEIGHT_IDX, WEIGHT_IDX)
from .eval_utils import avgerror, preprocess_depthmap, preprocess_targets, calculate_performance  # noqa: E402
from .uncertainty_utils import get_prediction_uncertainty_deepensemble  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

MIN_HEIGHT = 45
MAX_HEIGHT = 120
MAX_AGE = 1856.0

STUNTING_DIAGNOSIS = ["Not Stunted", "Moderately Stunted", "Severly Stunted"]
WASTING_DIAGNOSIS = ["Not Under-weight", "Moderately Under-weight", "Severly Under-weight"]


def process_image(data):
    img = tf.convert_to_tensor(data)
    img = tf.cast(img, tf.float32) * (1. / 256)
    img = tf.image.rot90(img, k=3)
    img = tf.image.resize(img, [240, 180])
    img = tf.expand_dims(img, axis=0)
    return img


def download_dataset(workspace: Workspace, dataset_name: str, dataset_path: str):
    logging.info("Accessing dataset...")
    if os.path.exists(dataset_path):
        return
    dataset = workspace.datasets[dataset_name]
    logging.info("Downloading dataset %s", dataset_name)
    dataset.download(target_path=dataset_path, overwrite=False)
    logging.info("Finished downloading %s", dataset_name)


def get_dataset_path(data_dir: Path, dataset_name: str):
    return str(data_dir / dataset_name)


def get_depthmap_files(paths: List[str]) -> List[str]:
    """Prepare the list of all the depthmap pickle files in dataset"""
    pickle_paths = []
    for path in paths:
        pickle_paths.extend(glob.glob(os.path.join(path, "**", "*.p")))
    return pickle_paths


def get_column_list(depthmap_path_list: List[str], prediction: np.array, DATA_CONFIG: Bunch):
    """Prepare the list of all artifact with its corresponding scantype, qrcode, target and prediction"""
    qrcode_list, scan_type_list, artifact_list, prediction_list, target_list = [], [], [], [], []

    for idx, path in enumerate(depthmap_path_list):
        loaded_tuple = pickle.load(open(path, "rb"))  # tuple can have 2 or 3 elements
        targets = loaded_tuple[1]
        targets = preprocess_targets(targets, DATA_CONFIG.TARGET_INDEXES)
        target = np.squeeze(targets)

        sub_folder_list = path.split('/')
        qrcode_list.append(sub_folder_list[-3])
        scan_type_list.append(sub_folder_list[-2])
        artifact_list.append(sub_folder_list[-1])
        prediction_list.append(prediction[idx])
        target_list.append(target)

    return qrcode_list, scan_type_list, artifact_list, prediction_list, target_list


def calculate_and_save_results(df_grouped: pd.DataFrame,
                               complete_name: str,
                               csv_out_fpath: str,
                               data_config: Bunch,
                               result_config: Bunch,
                               fct: Callable):
    """Calculate accuracies across the scantypes and save the final results table to the CSV file

    Args:
        df_grouped: dataframe grouped by 'qrcode' and 'scantype
        complete_name: e.g. 'q3-depthmap-plaincnn-height-100-95k-run_17'
        csv_out_fpath: CSV output path
        data_config: bunch containing data config
        result_config: bunch containing result config
        fct: Function to execute on inputs
    """
    dfs = []
    for code in data_config.CODES:
        df = fct(code, df_grouped, result_config)
        full_model_name = complete_name + CODE_TO_SCANTYPE[code]
        df.rename(index={0: full_model_name}, inplace=True)
        dfs.append(df)

    result = pd.concat(dfs)
    result.index.name = 'Model_Scantype'
    result = result.round(2)
    # Save the model results in csv file
    Path(csv_out_fpath).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_out_fpath, index=True)


def calculate_performance_sex(code: str, df_mae: pd.DataFrame, result_config: Bunch) -> pd.DataFrame:
    df_mae_filtered = df_mae.iloc[df_mae.index.get_level_values('scantype') == code]
    accuracy_list = calculate_accuracies(SEX_DICT.values(),
                                         df_mae_filtered,
                                         COLUMN_NAME_SEX,
                                         result_config.ACCURACY_MAIN_THRESH)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T
    df_out.columns = SEX_DICT.keys()
    return df_out


def calculate_performance_goodbad(code: str, df_mae: pd.DataFrame, result_config: Bunch) -> pd.DataFrame:
    df_mae_filtered = df_mae.iloc[df_mae.index.get_level_values('scantype') == code]
    accuracy_list = calculate_accuracies(GOODBAD_DICT.values(),
                                         df_mae_filtered,
                                         COLUMN_NAME_GOODBAD,
                                         result_config.ACCURACY_MAIN_THRESH)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T
    df_out.columns = GOODBAD_DICT.keys()
    return df_out


def calculate_performance_age(code: str, df_mae: pd.DataFrame, result_config: Bunch) -> pd.DataFrame:
    df_mae_filtered = df_mae.iloc[df_mae.index.get_level_values('scantype') == code]

    age_thresholds = result_config.AGE_BUCKETS
    age_buckets = list(zip(age_thresholds[:-1], age_thresholds[1:]))

    accuracy_list = calculate_accuracies_on_age_buckets(age_buckets,
                                                        df_mae_filtered,
                                                        COLUMN_NAME_AGE,
                                                        result_config.ACCURACY_MAIN_THRESH)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T

    df_out.columns = [f"{age_min} to {age_max}" for age_min, age_max in age_buckets]

    return df_out


def calculate_accuracies(values_to_select: List[float],
                         df: pd.DataFrame,
                         column_name: str,
                         accuracy_thresh: float) -> List[float]:
    """Take a dataframe with evaluation results and calculate cases above a threshold

    Args:
        values_to_select: Values that a certain column can have
        df: Needs to at least have to columns: 'error' and column_name
        column_name: Name of the column to select on
        accuracy_thresh: Error threshold

    Returns:
        A list of accuracies which has as many items as values_to_select
    """
    accuracy_list = []
    for idx in values_to_select:
        selection = (df[column_name] == idx)
        df_selected = df[selection]

        selection = (df_selected['error'] <= accuracy_thresh) & (df_selected['error'] >= -accuracy_thresh)
        accuracy = calc_accuracy_in_percent(num_all=len(df_selected), num_good=len(df_selected[selection]))
        accuracy_list.append(accuracy)
    return accuracy_list


def calculate_accuracies_on_age_buckets(age_buckets: List[Tuple[int]],
                                        df: pd.DataFrame,
                                        column_name: str,
                                        accuracy_thresh: float) -> List[float]:
    """Take a dataframe with evaluation results and calculate cases above a threshold

    Args:
        age_buckets: List of tuples where each tuple specifies a range: [age_min, age_max)
        df: Needs to at least have to columns: 'error' and column_name
        column_name: Name of the column to select on
        accuracy_thresh: Error threshold

    Returns:
        A list of accuracies which has as many items as values_to_select
    """
    accuracy_list = []
    for age_min_years, age_max_years in age_buckets:
        age_min = age_min_years * DAYS_IN_YEAR
        age_max = age_max_years * DAYS_IN_YEAR

        selection = (df[column_name] >= age_min) & (df[column_name] < age_max)
        df_selected = df[selection]

        selection = (df_selected['error'] <= accuracy_thresh) & (df_selected['error'] >= -accuracy_thresh)
        accuracy = calc_accuracy_in_percent(num_all=len(df_selected), num_good=len(df_selected[selection]))
        accuracy_list.append(accuracy)
    return accuracy_list


def calc_accuracy_in_percent(num_all: int, num_good: int) -> float:
    assert num_all >= num_good, f"num_all smaller than num_good: {num_good} < {num_all}"
    if num_all > 0:
        return num_good / num_all * 100
    return 0.


def draw_uncertainty_goodbad_plot(df_: pd.DataFrame, png_out_fpath: str):
    """Take all good samples and plot error distributions. Do the same for bad samples.

    Args:
        df: Dataframe with columns: goodbad and uncertainties
        png_out_fpath (str): File path where plot image will be saved
    """
    df = df_[df_.uncertainties.notna()]
    df_good = df[df[COLUMN_NAME_GOODBAD] == 1.0]
    df_bad = df[df[COLUMN_NAME_GOODBAD] == 0.0]

    good = list(df_good.uncertainties)
    bad = list(df_bad.uncertainties)

    bins = np.linspace(0, 10, 30)

    plt.hist(good, bins, alpha=0.5, label='good')
    plt.hist(bad, bins, alpha=0.5, label='bad')
    plt.title(f"Uncertainty plot: n_good={len(good)}, n_bad={len(bad)}")
    plt.xlabel("uncertainty in cm")
    plt.ylabel("occurrence count")
    plt.legend(loc='upper right')

    mean_good = float(df_good.uncertainties.mean())
    mean_bad = float(df_bad.uncertainties.mean())
    plt.axvline(mean_good, color='g', linestyle='dashed', linewidth=2)
    plt.axvline(mean_bad, color='r', linestyle='dashed', linewidth=2)

    plt.savefig(png_out_fpath)
    plt.close()


def draw_age_scatterplot(df_: pd.DataFrame, png_out_fpath: str):
    """Draw error over age scatterplot

    Args:
        df_: Dataframe with columns: qrcode, scantype, COLUMN_NAME_AGE, GT, predicted
        png_out_fpath: File path where plot image will be saved
    """
    df = df_[df_.scantype == '100'].groupby('qrcode').mean()
    df['error'] = df.apply(avgerror, axis=1).abs()
    plt.scatter(df[COLUMN_NAME_AGE], df['error'], s=2)
    plt.grid()
    plt.title("Per-scan Error over Age")
    plt.xlabel("age")
    plt.ylabel("error")
    axes = plt.gca()
    axes.set_xlim([0, 2500])
    axes.set_ylim([0, 5])
    Path(png_out_fpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_out_fpath)
    plt.close()


def draw_uncertainty_scatterplot(df: pd.DataFrame, png_out_fpath: str):
    """Draw error over age scatterplot

    Args:
        df: Dataframe with columns: qrcode, scantype, COLUMN_NAME_AGE, GT, predicted
        png_out_fpath: File path where plot image will be saved
    """
    df['error'] = df.apply(avgerror, axis=1).abs()
    plt.scatter(df['error'], df['uncertainties'], s=2)
    plt.grid()

    correlation, _ = pearsonr(df['error'], df['uncertainties'])
    logging.info("correlation: %d", correlation)

    plt.title(f"Per-scan sample artifact: Error over uncertainty (correlation={correlation:.3})")
    plt.xlabel("error")
    plt.ylabel("uncertainty (stdev of MC Dropout)")
    axes = plt.gca()
    axes.set_xlim([0, 5])
    axes.set_ylim([0, 10])
    Path(png_out_fpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_out_fpath)
    plt.close()


def draw_stunting_diagnosis(df: pd.DataFrame, png_out_fpath: str):
    """Draw stunting Confusion Matrix

    Args:
        df: Dataframe with columns: qrcode, scantype, COLUMN_NAME_AGE, GT, predicted
        png_out_fpath: File path where plot image will be saved
    """
    df = parallelize_dataframe(df, calculate_zscore_lhfa)
    actual = np.where(df['Z_actual'].values < -3, 'Severly Stunted',
                      np.where(df['Z_actual'].values > -2, 'Not Stunted', 'Moderately Stunted'))
    predicted = np.where(df['Z_predicted'].values < -3, 'Severly Stunted',
                         np.where(df['Z_predicted'].values > -2, 'Not Stunted', 'Moderately Stunted'))
    data = confusion_matrix(actual, predicted)
    draw_confusion_matrix(data, png_out_fpath, STUNTING_DIAGNOSIS, 'Stunting Diagnosis')


def calculate_zscore_lhfa(df):
    """lhfa: length/height for age"""
    cal = Calculator()

    def _calc_score(age_in_days, height, sex):
        if MIN_HEIGHT < height <= MAX_HEIGHT and age_in_days <= MAX_AGE:
            return cal.zScore_lhfa(age_in_days=age_in_days, sex=sex, height=height)

    def _fct(row):
        return _calc_score(age_in_days=int(row[COLUMN_NAME_AGE]),
                           sex='M' if row[COLUMN_NAME_SEX] == SEX_DICT['male'] else 'F', height=row['GT'])

    def _fct2(row):
        return _calc_score(age_in_days=int(row[COLUMN_NAME_AGE]),
                           sex='M' if row[COLUMN_NAME_SEX] == SEX_DICT['male'] else 'F', height=row['predicted'])

    df['Z_actual'] = df.apply(_fct, axis=1)
    df['Z_predicted'] = df.apply(_fct2, axis=1)
    return df


def draw_wasting_diagnosis(df: pd.DataFrame, png_out_fpath: str):
    """Draw wasting Confusion Matrix

    Args:
        df_: Dataframe with columns: qrcode, scantype, COLUMN_NAME_AGE, GT, predicted
        png_out_fpath: File path where plot image will be saved
    """
    df = parallelize_dataframe(df, calculate_zscore_wfa)
    actual = np.where(df['Z_actual'].values < -3, 'Severly Under-weight',
                      np.where(df['Z_actual'].values > -2, 'Not Under-weight', 'Moderately Under-weight'))
    predicted = np.where(df['Z_predicted'].values < -3, 'Severly Under-weight',
                         np.where(df['Z_predicted'].values > -2, 'Not Under-weight', 'Moderately Under-weight'))
    data = confusion_matrix(actual, predicted)
    draw_confusion_matrix(data, png_out_fpath, WASTING_DIAGNOSIS, 'Wasting Diagnosis')


def calculate_zscore_wfa(df):
    """Weight for age"""
    cal = Calculator()

    def utils(age_in_days, weight, sex):
        if age_in_days <= MAX_AGE:
            return cal.zScore_wfa(age_in_days=age_in_days, sex=sex, weight=weight)

    df['Z_actual'] = df.apply(
        lambda row: utils(age_in_days=int(row[COLUMN_NAME_AGE]),
                          sex='M' if row[COLUMN_NAME_SEX] == SEX_DICT['male'] else 'F', weight=row['GT']),
        axis=1)
    df['Z_predicted'] = df.apply(
        lambda row: utils(age_in_days=int(row[COLUMN_NAME_AGE]),
                          sex='M' if row[COLUMN_NAME_SEX] == SEX_DICT['male'] else 'F', weight=row['predicted']),
        axis=1)

    return df


def draw_confusion_matrix(data, png_out_fpath, display_labels, title):
    T, FP, FN = calculate_percentage_confusion_matrix(data)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    disp = ConfusionMatrixDisplay(confusion_matrix=data, display_labels=display_labels)
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    s = f"True: {round(T, 2)} False Positive: {round(FP, 2)} False Negative: {round(FN, 2)}"
    plt.text(0.5, 0.5, s, size=10, bbox=dict(boxstyle="square", facecolor='white'))
    ax.set_title(title)
    Path(png_out_fpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_out_fpath)
    plt.close()


def parallelize_dataframe(df, calculate_confusion_matrix, n_cores=8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(calculate_confusion_matrix, df_split))
    pool.close()
    pool.join()
    return df


def calculate_percentage_confusion_matrix(data):
    T1, FP1, FP2, FN1, T2, FP3, FN2, FN3, T3 = data.ravel()
    Total = sum(data.ravel())
    T = round(((T1 + T2 + T3) / Total) * 100, 2)
    FP = round(((FP1 + FP2 + FP3) / Total) * 100, 2)
    FN = round(((FN1 + FN2 + FN3) / Total) * 100, 2)
    return T, FP, FN


def get_model_path(MODEL_CONFIG: Bunch) -> str:
    if MODEL_CONFIG.NAME.endswith(".h5"):
        return MODEL_CONFIG.NAME
    if MODEL_CONFIG.NAME.endswith(".ckpt"):
        return os.path.join(MODEL_CONFIG.INPUT_LOCATION, MODEL_CONFIG.NAME)
    raise NameError(f"{MODEL_CONFIG.NAME}'s path extension not supported")


def download_model(workspace, experiment_name, run_id, input_location, output_location):
    """Download the pretrained model

    Args:
         workspace: workspace to access the experiment
         experiment_name: Name of the experiment in which model is saved
         run_id: Run Id of the experiment in which model is pre-trained
         input_location: Input location in a RUN Id
         output_location: Location for saving the model
    """
    experiment = Experiment(workspace=workspace, name=experiment_name)
    # Download the model on which evaluation need to be done
    run = Run(experiment, run_id=run_id)
    if input_location.endswith(".h5"):
        run.download_file(input_location, output_location)
    elif input_location.endswith(".ckpt"):
        run.download_files(prefix=input_location, output_directory=output_location)
    else:
        raise NameError(f"{input_location}'s path extension not supported")

    logging.info("Successfully downloaded model")


def filter_dataset_according_to_standing_lying(paths_evaluation, standing):
    new_paths_evaluation = []
    exc = []
    for p in paths_evaluation:
        _depthmap, _targets, image = pickle.load(open(p, "rb"))
        try:
            image = process_image(image)
            if standing.predict(image) > .9:
                new_paths_evaluation.append(p)
        except ValueError:
            exc.append(image)
    return new_paths_evaluation


def get_prediction(model_path: str, dataset_evaluation: tf.data.Dataset, DATA_CONFIG) -> np.array:
    """Perform the prediction on the dataset with the given model.

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
    logging.info("Total time for uncertainty prediction experiment: %.2f sec", end - start)

    prediction_list = np.squeeze(predictions)
    return prediction_list


def get_predictions_from_multiple_models(model_paths: list, dataset_evaluation: tf.data.Dataset, DATA_CONFIG) -> list:
    prediction_list_one = []
    for model_index, model_path in enumerate(model_paths):
        logging.info(f"Model {model_index + 1}/{len(model_paths)}")
        prediction_list_one += [get_prediction(model_path, dataset_evaluation, DATA_CONFIG)]
        logging.info("Prediction made by model on the depthmaps...")
    prediction_list_one = np.array(prediction_list_one)
    prediction_list_one = np.mean(prediction_list_one, axis=0)
    return prediction_list_one


def tf_load_pickle(path, max_value, DATA_CONFIG):
    """Utility to load the depthmap (may include RGB) pickle file"""
    def py_load_pickle(path, max_value):
        loaded_tuple = pickle.load(open(path.numpy(), "rb"))  # tuple can have 2 or 3 elements
        depthmap = loaded_tuple[0]
        targets = loaded_tuple[1]
        depthmap = preprocess_depthmap(depthmap)
        depthmap = depthmap / max_value
        depthmap = tf.image.resize(depthmap, (DATA_CONFIG.IMAGE_TARGET_HEIGHT, DATA_CONFIG.IMAGE_TARGET_WIDTH))
        targets = preprocess_targets(targets, DATA_CONFIG.TARGET_INDEXES)
        return depthmap, targets

    depthmap, targets = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
    depthmap.set_shape((DATA_CONFIG.IMAGE_TARGET_HEIGHT, DATA_CONFIG.IMAGE_TARGET_WIDTH, 1))
    targets.set_shape((len(DATA_CONFIG.TARGET_INDEXES,)))
    return path, depthmap, targets


def prepare_sample_dataset(df_sample, dataset_path, DATA_CONFIG):
    df_sample['artifact_path'] = df_sample.apply(
        lambda x: f"{dataset_path}/scans/{x['qrcode']}/{x['scantype']}/{x['artifact']}", axis=1)
    paths_evaluation = list(df_sample['artifact_path'])
    dataset_sample = tf.data.Dataset.from_tensor_slices(paths_evaluation)
    dataset_sample = dataset_sample.map(
        lambda path: tf_load_pickle(path, DATA_CONFIG.NORMALIZATION_VALUE, DATA_CONFIG)
    )
    dataset_sample = dataset_sample.map(lambda _path, depthmap, targets: (depthmap, targets))
    dataset_sample = dataset_sample.cache()
    dataset_sample = dataset_sample.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset_sample


class Evaluation:
    def __init__(self, model_config: Bunch, model_base_dir: Path, dataset_path: str) -> None:
        self.model_config = model_config
        self.model_base_dir = model_base_dir
        self.dataset_path = dataset_path
        self._input_location = os.path.join(self.model_config.INPUT_LOCATION, self.model_config.NAME)

    def get_the_model_path(self, workspace: Workspace):
        logging.info(f"Model will download from '{self._input_location}' to '{self.model_base_dir}'")
        download_model(workspace=workspace,
                       experiment_name=self.model_config.EXPERIMENT_NAME,
                       run_id=self.model_config.RUN_ID,
                       input_location=self._input_location,
                       output_location=self.model_base_dir)
        logging.info("Model was downloaded")
        self.model_path = self.model_base_dir / get_model_path(self.model_config)

    def get_the_qr_code_path(self) -> List[str]:
        dataset_path = os.path.join(self.dataset_path, "scans")
        logging.info('Dataset path: %s', dataset_path)
        logging.info('Getting QR-code paths...')
        qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
        logging.info('qrcode_paths: %d', len(qrcode_paths))
        assert len(qrcode_paths) != 0

        logging.info('Paths for evaluation: \n\t' + '\n\t'.join(qrcode_paths))
        logging.info(len(qrcode_paths))
        return qrcode_paths

    def prepare_dataset(self,
                        qrcode_paths: List[str],
                        DATA_CONFIG: Bunch,
                        FILTER_CONFIG: Bunch) -> Tuple[tf.data.Dataset, List[str]]:
        # Get depthmaps
        logging.info("Getting Depthmap paths...")
        paths_evaluation = get_depthmap_files(qrcode_paths)
        del qrcode_paths

        logging.info("Using %d artifact files for evaluation.", len(paths_evaluation))

        new_paths_evaluation = paths_evaluation

        if FILTER_CONFIG is not None and FILTER_CONFIG.IS_ENABLED:
            standing = load_model(FILTER_CONFIG.NAME)
            new_paths_evaluation = filter_dataset_according_to_standing_lying(paths_evaluation, standing)

        logging.info("Creating dataset for training.")
        paths = new_paths_evaluation
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        dataset_norm = dataset.map(
            lambda path: tf_load_pickle(path, DATA_CONFIG.NORMALIZATION_VALUE, DATA_CONFIG)
        )

        # filter goodbad==delete
        if GOODBAD_IDX in DATA_CONFIG.TARGET_INDEXES:
            goodbad_index = DATA_CONFIG.TARGET_INDEXES.index(GOODBAD_IDX)
            dataset_norm = dataset_norm.filter(
                lambda _path, _depthmap, targets: targets[goodbad_index] != GOODBAD_DICT['delete'])

        dataset_norm = dataset_norm.cache()
        dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
        temp_dataset_evaluation = dataset_norm
        del dataset_norm
        logging.info("Created dataset for training.")

        # Update new_paths_evaluation after filtering
        dataset_paths = temp_dataset_evaluation.map(lambda path, _depthmap, _targets: path)
        list_paths = list(dataset_paths.as_numpy_iterator())
        new_paths_evaluation = [x.decode() for x in list_paths]

        dataset_evaluation = temp_dataset_evaluation.map(lambda _path, depthmap, targets: (depthmap, targets))
        del temp_dataset_evaluation
        return dataset_evaluation, new_paths_evaluation

    def get_prediction_(self, model_path: Path, dataset_evaluation: tf.data.Dataset, DATA_CONFIG: Bunch) -> np.array:
        return get_prediction(model_path, dataset_evaluation, DATA_CONFIG)

    def prepare_dataframe(self, new_paths_evaluation, prediction_list_one, DATA_CONFIG, RESULT_CONFIG):
        qrcode_list, scantype_list, artifact_list, prediction_list, target_list = get_column_list(
            new_paths_evaluation, prediction_list_one, DATA_CONFIG)

        df = pd.DataFrame({
            'qrcode': qrcode_list,
            'artifact': artifact_list,
            'scantype': scantype_list,
            'GT': target_list if target_list[0].shape == tuple() else [el[0] for el in target_list],
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

        logging.info("df.shape: %s", df.shape)
        return df

    def evaluate(self,
                 df: pd.DataFrame,
                 DATA_CONFIG: Bunch,
                 RESULT_CONFIG: Bunch,
                 EVAL_CONFIG: Bunch,
                 OUTPUT_CSV_PATH: str,
                 descriptor: str):
        df_grouped = df.groupby(['qrcode', 'scantype']).mean()
        logging.info("Mean Avg Error: %s", df_grouped)

        df_grouped['error'] = df_grouped.apply(avgerror, axis=1)

        csv_fpath = f"{OUTPUT_CSV_PATH}/{descriptor}.csv"
        logging.info("Calculate and save the results to %s", csv_fpath)
        calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_fpath,
                                   DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance)

        sample_csv_fpath = f"{OUTPUT_CSV_PATH}/inaccurate_scans_{descriptor}.csv"
        df_grouped.to_csv(sample_csv_fpath, index=True)

        if 'AGE_BUCKETS' in RESULT_CONFIG.keys():
            csv_fpath = f"{OUTPUT_CSV_PATH}/age_evaluation_{descriptor}.csv"
            logging.info("Calculate and save age results to %s", csv_fpath)
            calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_fpath,
                                       DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance_age)
            png_fpath = f"{OUTPUT_CSV_PATH}/age_evaluation_scatter_{descriptor}.png"
            logging.info("Calculate and save scatterplot results to %s", png_fpath)
            draw_age_scatterplot(df, png_fpath)

        if (HEIGHT_IDX in DATA_CONFIG.TARGET_INDEXES
                and AGE_IDX in DATA_CONFIG.TARGET_INDEXES
                and descriptor != self.model_config.EXPERIMENT_NAME):
            png_fpath = f"{OUTPUT_CSV_PATH}/stunting_diagnosis_{descriptor}.png"
            logging.info("Calculate zscores and save confusion matrix results to %s", png_fpath)
            start = time.time()
            draw_stunting_diagnosis(df, png_fpath)
            end = time.time()
            logging.info("Total time for Calculate zscores and save confusion matrix: %.2f", end - start)

        if (WEIGHT_IDX in DATA_CONFIG.TARGET_INDEXES
                and AGE_IDX in DATA_CONFIG.TARGET_INDEXES
                and descriptor != self.model_config.EXPERIMENT_NAME):
            png_fpath = f"{OUTPUT_CSV_PATH}/wasting_diagnosis_{descriptor}.png"
            logging.info("Calculate and save wasting confusion matrix results to %s", png_fpath)
            start = time.time()
            draw_wasting_diagnosis(df, png_fpath)
            end = time.time()
            logging.info("Total time for Calculate zscores and save wasting confusion matrix: %.2f", end - start)

        if SEX_IDX in DATA_CONFIG.TARGET_INDEXES:
            csv_fpath = f"{OUTPUT_CSV_PATH}/sex_evaluation_{descriptor}.csv"
            logging.info("Calculate and save sex results to %s", csv_fpath)
            calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_fpath,
                                       DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance_sex)
        if GOODBAD_IDX in DATA_CONFIG.TARGET_INDEXES:
            csv_fpath = f"{OUTPUT_CSV_PATH}/goodbad_evaluation_{descriptor}.csv"
            logging.info("Calculate performance on bad/good scans and save results to %s", csv_fpath)
            calculate_and_save_results(df_grouped, EVAL_CONFIG.NAME, csv_fpath,
                                       DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance_goodbad)


class EnsembleEvaluation(Evaluation):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_the_model_path(self, workspace: Workspace):
        """Get multiple model paths"""
        for run_id in self.model_config.RUN_IDS:
            logging.info(f"Downloading run {run_id}")
            download_model(
                workspace=workspace,
                experiment_name=self.model_config.EXPERIMENT_NAME,
                run_id=run_id,
                input_location=self._input_location,
                output_location=self.model_base_dir / run_id
            )
        model_paths = glob.glob(os.path.join(self.model_base_dir, "*"))
        model_paths = [p for p in model_paths if os.path.isdir(p)]
        model_paths = [p for p in model_paths if p.split("/")[-1].startswith(self.model_config.EXPERIMENT_NAME)]
        model_paths = [os.path.join(p, self.model_config.INPUT_LOCATION, self.model_config.NAME) for p in model_paths]
        logging.info(f"Models paths ({len(model_paths)}):")
        logging.info("\t" + "\n\t".join(model_paths))
        self.model_paths = model_paths

    def get_prediction_(self,
                        model_paths: List[Path],
                        dataset_evaluation: tf.data.Dataset,
                        DATA_CONFIG: Bunch) -> np.array:
        return get_predictions_from_multiple_models(model_paths, dataset_evaluation, DATA_CONFIG)

    def evaluate(self,
                 df: pd.DataFrame,
                 DATA_CONFIG: Bunch,
                 RESULT_CONFIG: Bunch,
                 EVAL_CONFIG: Bunch,
                 OUTPUT_CSV_PATH: str,
                 descriptor: str):
        super().evaluate(df, DATA_CONFIG, RESULT_CONFIG, EVAL_CONFIG, OUTPUT_CSV_PATH, descriptor)

        if not RESULT_CONFIG.USE_UNCERTAINTY:
            return

        assert GOODBAD_IDX in DATA_CONFIG.TARGET_INDEXES
        assert COLUMN_NAME_GOODBAD in df

        # Sample one artifact per scan (qrcode, scantype combination)
        df_sample = df.groupby(['qrcode', 'scantype']).apply(lambda x: x.sample(1))

        # Prepare uncertainty prediction on these artifacts
        dataset_sample = prepare_sample_dataset(df_sample, self.dataset_path, DATA_CONFIG)

        # Predict uncertainty
        uncertainties = get_prediction_uncertainty_deepensemble(self.model_paths, dataset_sample)

        assert len(df_sample) == len(uncertainties)
        df_sample['uncertainties'] = uncertainties

        png_fpath = f"{OUTPUT_CSV_PATH}/uncertainty_distribution.png"
        draw_uncertainty_goodbad_plot(df_sample, png_fpath)

        df_sample_100 = df_sample.iloc[df_sample.index.get_level_values('scantype') == '100']
        png_fpath = f"{OUTPUT_CSV_PATH}/uncertainty_code100_distribution.png"
        draw_uncertainty_goodbad_plot(df_sample_100, png_fpath)

        png_fpath = f"{OUTPUT_CSV_PATH}/uncertainty_scatter_distribution.png"
        draw_uncertainty_scatterplot(df_sample, png_fpath)

        # Filter for scans with high certainty and calculate their accuracy/results
        df_sample['error'] = df_sample.apply(avgerror, axis=1).abs()
        df_sample_better_threshold = df_sample[df_sample['uncertainties'] < RESULT_CONFIG.UNCERTAINTY_THRESHOLD_IN_CM]
        csv_fpath = f"{OUTPUT_CSV_PATH}/uncertainty_smaller_than_{RESULT_CONFIG.UNCERTAINTY_THRESHOLD_IN_CM}cm.csv"
        logging.info("Uncertainty: For more certain than %.2f cm, calculate and save the results to %s",
                     RESULT_CONFIG.UNCERTAINTY_THRESHOLD_IN_CM, csv_fpath)
        calculate_and_save_results(df_sample_better_threshold, EVAL_CONFIG.NAME, csv_fpath,
                                   DATA_CONFIG, RESULT_CONFIG, fct=calculate_performance)

# class MultiartifactEvaluation(Evaluation):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
