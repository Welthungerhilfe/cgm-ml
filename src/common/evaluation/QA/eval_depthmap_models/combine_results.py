import argparse
import pandas as pd
import logging

from glob2 import glob


OUTPUT_FILE_NAME = 'evaluated_models_result.csv'


def combine_model_results(csv_file_list, output_path):
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
    result_csv = f"{output_path}/{OUTPUT_FILE_NAME}"
    final_result.to_csv(result_csv, index=True)


if __name__ == "__main__":
    paths = {
        'height': 'outputs/height',
        'weight': 'outputs/weight'
    }

    def validate_arg(args_string):
        """
        Function to validate the passing args

        Args:
            arg_string (string): input argument

        Raises:
            argparse.ArgumentTypeError: error to throw if the value is not valid
        """
        value = args_string.lower()
        if value not in paths.keys():
            raise argparse.ArgumentTypeError("%s is an invalid argument value" % args_string)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_measurement",
        default="height",
        type=validate_arg,
        help="defining models usage for the measuring height or weight")
    args = parser.parse_args()
    model_measurement_type = args.model_measurement
    model_measurement_type = model_measurement_type.lower()
    result_path = paths.get(model_measurement_type)
    csv_path = f"{result_path}/*.csv"
    csv_files = glob(csv_path)
    combine_model_results(csv_files, result_path)
