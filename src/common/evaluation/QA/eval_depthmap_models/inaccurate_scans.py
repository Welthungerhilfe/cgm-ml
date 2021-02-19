import argparse
import pandas as pd
import logging
from typing import List

from glob2 import glob

OUTPUT_FILE_NAME = 'evaluated_models_result.csv'

ACC = 2

def merge_qrc(row):
    scans = str(row['qrcode'])+'_'+str(row['scantype'])
    return scans

def filter_scans(dataframe: pd.DataFrame,accuracy: int):
    error = dataframe[(dataframe['error'] >= accuracy) | (dataframe['error'] <= -accuracy)]
    return error
    
def inaccurate_scans(csv_file_list: List[str]):
    """Function to combine the models resultant csv files into a single file

    Args:
        csv_file_list: list containing absolute path of csv file
        output_path: target folder path where to save result csv file
    """
    if len(csv_file_list) <= 0:
        logging.warning("No csv files found in output directory to combine")
        return
    count = 0
    for file in csv_file_list:
        result_list = pd.read_csv(file,index=False)
        grouped_result = result_list.groupby(['qrcode','scantype'],as_index=False).mean()
        accuracy_df = filter_scans(grouped_result,ACC)
        accuracy_df['name'] = accuracy_df.apply(merge_qrc,axis=1)
        csv_file = f"./outputs/{count}.csv"
        count+=1
        
    

if __name__ == "__main__":
    paths = {
        'height': 'outputs/height',
        'weight': 'outputs/weight'
    }
    
    csv_path = f"{result_path}/*_inaccurate_scan.csv"
    csv_files = glob(csv_path)
    inaccurate_scans(csv_files)
