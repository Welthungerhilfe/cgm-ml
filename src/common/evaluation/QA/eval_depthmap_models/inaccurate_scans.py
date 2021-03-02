import pandas as pd
import logging

from glob2 import glob
from pathlib import Path

ACCURACY_THRESHOLD = 2
CSV_PATH = "./outputs/**/*_scans.csv"
FIGURE_NAME = 'common_inaccurate_scans.png'


def merge_qrc(row):
    """
    Function to combine qrcodes with scantypes
    """
    scans = str(row['qrcode']) + '_' + str(row['scantype'])
    return scans


def filter_scans(dataframe: pd.DataFrame, accuracy: int) -> pd.DataFrame:
    """
    Function that filter dataframe for the fiven accuracy number
    """
#     error = dataframe[(dataframe['error'] >= accuracy) | (dataframe['error'] <= -accuracy)]
    error = dataframe[dataframe['error'].abs() >= accuracy]
    return error


def frame_to_set(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Function to convert dataframe column to list
    """
    return set(dataframe['name'].to_list())


def calculate_union(set1: set, set2: set) -> set:
    """
    Function to calculate union of two sets
    """
    union_set = set1.union(set2)
    return union_set


def calculate_intersection(set1: set, set2: set) -> set:
    """
    Function to calculate intersection of two sets
    """
    intersection_set = set1.intersection(set2)
    return intersection_set


def extract_model_name(path_name) -> str:
    """
    Function to extract the model name from the path. 
    """
    assert path_name.endswith('.csv')
    model_name = Path(path_name).resolve().stem
    return model_name


def inaccurate_scans(file) -> pd.DataFrame:  # noqa: E402
    """                                
    Function to combine the models resultant csv files into a single file
    Args:
        csv_file_list: list containing absolute path of csv file
        output_path: target folder path where to save result csv file
    Returns: 
        panda dataframe: dataframe with filter results based on accuracy
    """
    result_list = pd.read_csv(file)
    grouped_result = result_list.groupby(['qrcode', 'scantype'], as_index=False).mean()
    accuracy_df = filter_scans(grouped_result, ACCURACY_THRESHOLD)
    accuracy_df['name'] = accuracy_df.apply(merge_qrc, axis=1)
    frame_set = frame_to_set(accuracy_df)
    return frame_set


if __name__ == "__main__":

    csv_files = glob(CSV_PATH)
    if len(csv_files) != 2:
        logging.warning("path contains 0 or more than 2 csv files")
    scan_sets = []
    for file in csv_files:
        frame_set = inaccurate_scans(file)
        scan_sets.append(frame_set)

    union_set = calculate_union(scan_sets[0], scan_sets[1])
    intersection_set = calculate_intersection(scan_sets[0], scan_sets[1])
    percentage = (len(intersection_set) / len(union_set)) * 100
    model_name = [[extract_model_name(csv_files[0]), extract_model_name(
        csv_files[1]), percentage, len(union_set), len(intersection_set)]]
    columns = ['model_1', 'model_2', 'overlap_percentage', 'Total_scanstype', 'intersection']
    frame = pd.DataFrame(model_name, columns=columns)
    frame.to_csv('inaccurate_scan_report.csv')
