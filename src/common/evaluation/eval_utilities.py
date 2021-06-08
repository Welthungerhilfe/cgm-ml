import logging
import logging.config
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Tuple

import glob2 as glob
import numpy as np
import pandas as pd
import tensorflow as tf
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
    GOODBAD_DICT, SEX_DICT)
from .eval_utils import avgerror, preprocess_targets  # noqa: E402


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


def get_column_list(depthmap_path_list: List[str], prediction: np.array, DATA_CONFIG: Bunch, FILTER_CONFIG: Bunch):
    """Prepare the list of all artifact with its corresponding scantype, qrcode, target and prediction"""
    qrcode_list, scan_type_list, artifact_list, prediction_list, target_list = [], [], [], [], []

    for idx, path in enumerate(depthmap_path_list):
        if FILTER_CONFIG is not None:
            _, targets, _ = pickle.load(open(path, "rb"))  # For filter(contains RGBs) dataset
        else:
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
        return f"{MODEL_CONFIG.INPUT_LOCATION}/{MODEL_CONFIG.NAME}"
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
