import argparse
import pandas as pd
import logging
import os
from importlib import import_module
from glob2 import glob


OUTPUT_FILE_NAME = 'evaluated_models_result.csv'


def combine_model_results(csv_file_list,output_path):
    """
    Function to combine the models resultant csv files into a single file

    Args:
        csv_file_list (list): list containing absolute path of csv file
        output_path('str'): target folder path where to save result csv file
    """
    if len(csv_file_list) <= 0:
        logging.warning("No csv files found in output directory to combine")
        return
    result_list = []
    for results in csv_files:
        read_csv_file = pd.read_csv(results, index_col=0)
        result_list.append(read_csv_file)
    final_result = pd.concat(result_list, axis=0)
    final_result = final_result.rename_axis("Model")
    final_result = final_result.round(2)
    result_csv = "{}{}{}".format(output_path,'/',OUTPUT_FILE_NAME)
    final_result.to_csv(result_csv, index=True)


if __name__ == "__main__":
    PATHS = {
        'height' : 'outputs/height',
        'weight' : 'outputs/weight'
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_measurement", default="height", help="defining models usage for the measuring height or weight ")
    args = parser.parse_args()
    model_measurement_type = args.model_measurement
    model_measurement_type = model_measurement_type.lower()
    result_path = PATHS.get(model_measurement_type)
    csv_path = "{}{}".format(result_path,'/*.csv')
    csv_files = glob(csv_path)
    combine_model_results(csv_files,result_path)
