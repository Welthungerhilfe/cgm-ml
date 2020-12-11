from bunch import Bunch
import datetime
import os
from pathlib import Path
import pickle
from typing import List

from azureml.core import Experiment, Run, Workspace
import glob2 as glob
import numpy as np
import pandas as pd


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
    if targets_indices is not None:
        targets = targets[targets_indices]
    return targets.astype("float32")


def get_depthmap_files(paths: List[str]) -> List[str]:
    '''
    Prepare the list of all the depthmap pickle files in dataset
    '''
    pickle_paths = []
    for path in paths:
        pickle_paths.extend(glob.glob(os.path.join(path, "**", "*.p")))
    return pickle_paths


def get_column_list(depthmap_path_list: List[str], prediction: np.array, DATA_CONFIG: Bunch):
    """Prepare the list of all artifact with its corresponding scantype, qrcode, target and prediction"""
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


def calculate_performance(code: str, df_mae: pd.DataFrame, RESULT_CONFIG: Bunch) -> pd.DataFrame:
    """For each scantype, calculate the performance of the model across all error margin

    Args:
        code: e.g. '100'
        df_mae: [description]
        RESULT_CONFIG: [description]

    Returns:
        [description]
    """
    df_mae_filtered = df_mae.iloc[df_mae.index.get_level_values('scantype') == code]
    accuracy_list = []
    for acc in RESULT_CONFIG.ACCURACIES:
        good_predictions = df_mae_filtered[(df_mae_filtered['error'] <= acc) & (df_mae_filtered['error'] >= -acc)]
        if len(df_mae_filtered):
            accuracy = len(good_predictions) / len(df_mae_filtered) * 100
        else:
            accuracy = 0.
        accuracy_list.append(accuracy)
    df_out = pd.DataFrame(accuracy_list)
    df_out = df_out.T
    df_out.columns = RESULT_CONFIG.ACCURACIES
    return df_out


def calculate_and_save_results(MAE: pd.DataFrame, complete_name: str, CSV_OUT_PATH: str, DATA_CONFIG: Bunch, RESULT_CONFIG: Bunch, RUN_ID: str):
    """Calculate accuracies across the scantypes and save the final results table to the CSV file

    Args:
        MAE: dataframe
        complete_name: e.g. 'q3-depthmap-plaincnn-height-100-95k-run_17'
        CSV_OUT_PATH: CSV output path
        DATA_CONFIG: bunch containing data config
        RESULT_CONFIG: bunch containing result config
        RUN_ID: e.g. 'q3-depthmap-plaincnn-height-95k_1597988908_42c4ef33'
    """
    dfs = []
    for code in DATA_CONFIG.CODE_TO_SCANTYPE.keys():
        df = calculate_performance(code, MAE, RESULT_CONFIG)
        full_model_name = complete_name + DATA_CONFIG.CODE_TO_SCANTYPE[code]
        df.rename(index={0: full_model_name}, inplace=True)
        dfs.append(df)

    result = pd.concat(dfs)
    result.index.name = 'Model_Scantype'
    result = result.round(2)
    # Save the model results in csv file
    Path(CSV_OUT_PATH).mkdir(parents=True, exist_ok=True)
    csv_file = f"{CSV_OUT_PATH}/{RUN_ID}.csv"
    result.to_csv(csv_file, index=True)


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
    if input_location.endswith(".h5"):
        run.download_file(input_location, output_location)
    elif input_location.endswith(".ckpt"):
        run.download_files(prefix=input_location, output_directory=output_location)
    else:
        raise NameError(f"{input_location}'s path extension not supported")
    print("Successfully downloaded model")
