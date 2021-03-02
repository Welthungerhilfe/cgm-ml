import argparse
import pandas as pd
import logging
from typing import List

from glob2 import glob

OUTPUT_FILE_NAME = 'evaluated_models_result.csv'


def combine_model_results(csv_file_list: List[str], output_path: str):
    """Function to combine the models resultant csv files into a single file

    Args:
        csv_file_list: list containing absolute path of csv file
        output_path: target folder path where to save result csv file
    """
    if len(csv_file_list) <= 0:
        logging.warning("No csv files found in output directory to combine")
        return
    result_list = [pd.read_csv(results, index_col=0) for results in csv_file_list]
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

    def validate_arg(args_string: str) -> str:
        """Validate the passed arguments

        Args:
            arg_string: input argument

        Raises:
            argparse.ArgumentTypeError: error to throw if the value is not valid

        Returns: args_string
        """
        if args_string not in paths.keys():
            raise argparse.ArgumentTypeError(
                f"{args_string} is an invalid argument value. Valid options ared: {paths.keys()}")
        return args_string

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_measurement",
        default="height",
        type=validate_arg,
        help="defining models usage for the measuring height or weight")
    args = parser.parse_args()
    result_path = paths.get(args.model_measurement)
    csv_path = f"{result_path}/*.csv"
    csv_files = glob(csv_path)
    combine_model_results(csv_files, result_path)
