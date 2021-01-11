import copy
import datetime
import os
import pickle
from pathlib import Path
from typing import List, Callable

import glob2 as glob
import numpy as np
import pandas as pd
from azureml.core import Experiment, Run, Workspace
from bunch import Bunch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.python import keras

DAYS_IN_YEAR = 365

HEIGHT_IDX = 0
WEIGHT_IDX = 1
MUAC_IDX = 2
AGE_IDX = 3
SEX_IDX = 4
GOODBAD_IDX = 5

SEX_DICT = {'female': 0., 'male': 1.}
GOODBAD_DICT = {'bad': 0., 'good': 1.}

COLUMN_NAME_AGE = 'GT_age'
COLUMN_NAME_SEX = 'GT_sex'
COLUMN_NAME_GOODBAD = 'GT_goodbad'
CODE_TO_SCANTYPE = {
    '100': '_standingfront',
    '101': '_standing360',
    '102': '_standingback',
    '200': '_lyingfront',
    '201': '_lyingrot',
    '202': '_lyingback',
}


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


def download_dataset(workspace: Workspace, dataset_name: str, dataset_path: str):
    print("Accessing dataset...")
    if os.path.exists(dataset_path):
        return
    dataset = workspace.datasets[dataset_name]
    print(f"Downloading dataset {dataset_name}.. Current date and time: ",
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    dataset.download(target_path=dataset_path, overwrite=False)
    print(f"Finished downloading {dataset_name}, Current date and time: ",
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def get_dataset_path(data_dir: Path, dataset_name: str):
    return str(data_dir / dataset_name)


def preprocess_depthmap(depthmap):
    # TODO here be more code.
    return depthmap.astype("float32")


def preprocess_targets(targets, targets_indices):
    if SEX_IDX in targets_indices:
        targets[SEX_IDX] = SEX_DICT[targets[SEX_IDX]]
    if GOODBAD_IDX in targets_indices:
        try:
            targets[GOODBAD_IDX] = GOODBAD_DICT[targets[GOODBAD_IDX]]
        except KeyError:
            print(f"Key '{targets[GOODBAD_IDX]}' not found in GOODBAD_DICT")
            targets[GOODBAD_IDX] = 0.  # unknown target values will be categorized as 'bad'

    if targets_indices is not None:
        targets = targets[targets_indices]
    return targets.astype("float32")


def get_depthmap_files(paths: List[str]) -> List[str]:
    """Prepare the list of all the depthmap pickle files in dataset"""
    pickle_paths = []
    for path in paths:
        pickle_paths.extend(glob.glob(os.path.join(path, "**", "*.p")))
    return pickle_paths


def get_column_list(depthmap_path_list: List[str], prediction: np.array, data_config: Bunch):
    """Prepare the list of all artifact with its corresponding scantype, qrcode, target and prediction"""
    qrcode_list, scan_type_list, artifact_list, prediction_list, target_list = [], [], [], [], []

    for idx, path in enumerate(depthmap_path_list):
        _, targets = pickle.load(open(path, "rb"))
        targets = preprocess_targets(targets, data_config.TARGET_INDEXES)
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


def calculate_performance(code, df_mae, result_config):
    """For a specific scantype, calculate the performance of the model on each error margin
    Args:
        code: e.g. '100'
        df_mae: dataframe
        result_config: bunch containing result config
    Returns:
        dataframe, where each column describes a differnt accuracy, e.g.
                            0.2   0.4   0.6   1.0   1.2    2.0    2.5    3.0    4.0    5.0    6.0
                           20.0  20.0  40.0  80.0  80.0  100.0  100.0  100.0  100.0  100.0  100.0
    """
    df_mae_filtered = df_mae.iloc[df_mae.index.get_level_values('scantype') == code]
    accuracy_list = []
    for acc in result_config.ACCURACIES:
        good_predictions = df_mae_filtered[(df_mae_filtered['error'] <= acc) & (df_mae_filtered['error'] >= -acc)]
        if len(df_mae_filtered):
            accuracy = len(good_predictions) / len(df_mae_filtered) * 100
        else:
            accuracy = 0.
        accuracy_list.append(accuracy)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T
    df_out.columns = result_config.ACCURACIES
    return df_out


def calculate_and_save_results(df_grouped: pd.DataFrame, complete_name: str, csv_out_fpath: str, data_config: Bunch, result_config: Bunch, fct: Callable):
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
    accuracy_list = []
    accuracy_thresh = result_config.ACCURACY_MAIN_HEIGHT_THRESH
    for _, sex_id in SEX_DICT.items():
        selection = (df_mae_filtered[COLUMN_NAME_SEX] == sex_id)
        df = df_mae_filtered[selection]

        selection = (df['error'] <= accuracy_thresh) & (df['error'] >= -accuracy_thresh)
        good_predictions = df[selection]
        if len(df):
            accuracy = len(good_predictions) / len(df) * 100
        else:
            accuracy = 0.
        accuracy_list.append(accuracy)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T
    df_out.columns = SEX_DICT.keys()
    return df_out


def calculate_performance_goodbad(code: str, df_mae: pd.DataFrame, result_config: Bunch) -> pd.DataFrame:
    df_mae_filtered = df_mae.iloc[df_mae.index.get_level_values('scantype') == code]
    accuracy_list = []
    accuracy_thresh = result_config.ACCURACY_MAIN_HEIGHT_THRESH
    for _, goodbad_id in GOODBAD_DICT.items():
        selection = (df_mae_filtered[COLUMN_NAME_GOODBAD] == goodbad_id)
        df = df_mae_filtered[selection]

        selection = (df['error'] <= accuracy_thresh) & (df['error'] >= -accuracy_thresh)
        good_predictions = df[selection]
        if len(df):
            accuracy = len(good_predictions) / len(df) * 100
        else:
            accuracy = 0.
        accuracy_list.append(accuracy)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T
    df_out.columns = GOODBAD_DICT.keys()
    return df_out


def calculate_performance_age(code: str, df_mae: pd.DataFrame, result_config: Bunch) -> pd.DataFrame:
    df_mae_filtered = df_mae.iloc[df_mae.index.get_level_values('scantype') == code]
    accuracy_list = []
    accuracy_thresh = result_config.ACCURACY_MAIN_HEIGHT_THRESH
    age_thresholds = result_config.AGE_BUCKETS
    age_buckets = list(zip(age_thresholds[:-1], age_thresholds[1:]))
    for age_min_years, age_max_years in age_buckets:
        age_min = age_min_years * DAYS_IN_YEAR
        age_max = age_max_years * DAYS_IN_YEAR

        selection = (df_mae_filtered[COLUMN_NAME_AGE] >= age_min) & (df_mae_filtered[COLUMN_NAME_AGE] <= age_max)
        df = df_mae_filtered[selection]

        selection = (df['error'] <= accuracy_thresh) & (df['error'] >= -accuracy_thresh)
        good_predictions = df[selection]
        if len(df):
            accuracy = len(good_predictions) / len(df) * 100
        else:
            accuracy = 0.
        accuracy_list.append(accuracy)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T

    df_out.columns = [f"{age_min} to {age_max}" for age_min, age_max in age_buckets]

    return df_out


def draw_uncertainty_goodbad_plot(df_: pd.DataFrame, csv_out_fpath: str):
    """Take all good samples and plot error distributions. Do the same for bad samples.

    Args:
        df: Dataframe with columns: goodbad and uncertainties
        csv_out_fpath (str): File path where plot image will be saved
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

    plt.savefig(csv_out_fpath)
    plt.close()


def draw_age_scatterplot(df_: pd.DataFrame, csv_out_fpath: str):
    """Draw error over age scatterplot

    Args:
        df_: Dataframe with columns: qrcode, scantype, COLUMN_NAME_AGE, GT, predicted
        csv_out_fpath: File path where plot image will be saved
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
    Path(csv_out_fpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(csv_out_fpath)
    plt.close()


def get_model_path(MODEL_CONFIG: Bunch) -> str:
    if MODEL_CONFIG.NAME.endswith(".h5"):
        return MODEL_CONFIG.NAME
    elif MODEL_CONFIG.NAME.endswith(".ckpt"):
        return f"{MODEL_CONFIG.INPUT_LOCATION}/{MODEL_CONFIG.NAME}"
    raise NameError(f"{MODEL_CONFIG.NAME}'s path extension not supported")


def download_model(ws, experiment_name, run_id, input_location, output_location):
    """Download the pretrained model

    Args:
         ws: workspace to access the experiment
         experiment_name: Name of the experiment in which model is saved
         run_id: Run Id of the experiment in which model is pre-trained
         input_location: Input location in a RUN Id
         output_location: Location for saving the model
    """
    experiment = Experiment(workspace=ws, name=experiment_name)
    #Download the model on which evaluation need to be done
    run = Run(experiment, run_id=run_id)
    if input_location.endswith(".h5"):
        run.download_file(input_location, output_location)
    elif input_location.endswith(".ckpt"):
        run.download_files(prefix=input_location, output_directory=output_location)
    else:
        raise NameError(f"{input_location}'s path extension not supported")
    print("Successfully downloaded model")
