import pandas as pd
import logging

from glob2 import glob
from pathlib import Path

ACCURACY_THRESHOLD = 2
CSV_PATH = "./outputs/**/inaccurate_scans_*.csv"
REPORT_CSV = 'inaccurate_scan_report.csv'


def merge_qrc(row):
    """
    Function to combine qrcodes with scantypes
    """
    qrcode = row['qrcode']
    scantype = row['scantype']
    scans = f"{qrcode}_{scantype}"
    return scans


def filter_scans(dataframe: pd.DataFrame, accuracy: int) -> pd.DataFrame:
    """
    Function that filter dataframe for the fiven accuracy number
    """
    error = dataframe[dataframe['error'].abs() >= accuracy]
    return error


def frame_to_set(dataframe: pd.DataFrame) -> set:
    """
    Function to convert dataframe column to list
    """
    return set(dataframe['scan_code'].to_list())


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
    model_name = Path(path_name).stem
    return model_name


def calculate_inaccurate_scans(csv_filepath: str) -> set:
    """
    Function to combine the models resultant csv files into a single file
    """
    assert csv_filepath.endswith('.csv')
    result_list = pd.read_csv(csv_filepath)
    grouped_result = result_list.groupby(['qrcode', 'scantype'], as_index=False).mean()
    accuracy_df = filter_scans(grouped_result, ACCURACY_THRESHOLD)
    accuracy_df['scan_code'] = accuracy_df.apply(merge_qrc, axis=1)
    csv_name = csv_filepath.split('/')[-1]
    file_name = f"file_{csv_name}"
    accuracy_df.to_csv(file_name, index=False)
    frame_set = frame_to_set(accuracy_df)
    return frame_set


if __name__ == "__main__":
    csv_files = glob(CSV_PATH)
    if len(csv_files) != 2:
        logging.warning("path contains 0 or more than 2 csv files")

    scan_sets = [calculate_inaccurate_scans(filepath) for filepath in csv_files]
    union_set = calculate_union(scan_sets[0], scan_sets[1])
    inaccurate_scans_intersection = calculate_intersection(scan_sets[0], scan_sets[1])
    inaccurate_scans_intersection_ratio = (len(inaccurate_scans_intersection) / len(union_set)) * 100
    inaccurate_scan_data = [[extract_model_name(csv_files[0]), extract_model_name(
        csv_files[1]), inaccurate_scans_intersection_ratio, len(union_set), len(inaccurate_scans_intersection)]]
    columns = ['model_1', 'model_2', 'ratio_intersection_over_union',
               'number_of_union_of_inaccurate_scans', 'number_of_common_inaccurate_scans']
    frame = pd.DataFrame(inaccurate_scan_data, columns=columns)
    frame.to_csv(REPORT_CSV)
