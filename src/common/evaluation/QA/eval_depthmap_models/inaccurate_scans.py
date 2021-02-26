import argparse
import pandas as pd
import logging
from typing import List

from glob2 import glob
from matplotlib_venn import venn2   
from matplotlib import pyplot as plt
from pathlib import Path


ACC = 2

CSV_PATH = f"./outputs/**/*_scans.csv"

FIGURE_NAME = 'common_inaccurate_scans.png'

def merge_qrc(row):
    """
    Function to combine qrcodes with scantypes
    """
    scans = str(row['qrcode'])+'_'+str(row['scantype'])
    return scans

def filter_scans(dataframe: pd.DataFrame,accuracy: int):
    """
    Function that filter dataframe for the fiven accuracy number
    """
    error = dataframe[(dataframe['error'] >= accuracy) | (dataframe['error'] <= -accuracy)]
    return error

def frame_to_set(dataframe: pd.DataFrame):
    """
    Function to convert dataframe column to list
    """
    return set(dataframe['name'].to_list())

def calculate_union(set1:set,set2:set):
    """
    Function to calculate union of two sets
    """
    union_set = set1.union(set2)
    return union_set

def calculate_intersection(set1:set,set2:set):
    """
    Function to calculate intersection of two sets
    """
    intersection_set = set1.intersection(set2)
    return intersection_set

def extract_model_name(path_name):
    """
    Function to extract the model name from the path. 
    """
    assert path_name.endswith('.csv')
    model_name = Path(path_name).resolve().stem
    return model_name
    

def inaccurate_scans(file: List[str]):
    """
    Function to combine the models resultant csv files into a single file
    Args:
        csv_file_list: list containing absolute path of csv file
        output_path: target folder path where to save result csv file
    """
    file_path = file.rsplit('/',1)[0]
    result_list = pd.read_csv(file)
    grouped_result = result_list.groupby(['qrcode','scantype'],as_index=False).mean()
    accuracy_df = filter_scans(grouped_result,ACC)
    accuracy_df['name'] = accuracy_df.apply(merge_qrc,axis=1)
    print("Length of the accuracy_df:",len(accuracy_df))
    frame_set = frame_to_set(accuracy_df)
    return frame_set

if __name__ == "__main__": 
    
    csv_files = glob(CSV_PATH)
    if len(csv_files) !=2:
        logging.warning("path contains 0 or more than 2 csv files")
    scan_sets = []
    for file in csv_files:
        frame_set = inaccurate_scans(file)
        scan_sets.append(frame_set)
        
    Union_set = calculate_union(scan_sets[0],scan_sets[1])
    Intersection_set = calculate_intersection(scan_sets[0],scan_sets[1])
    percentage  = (len(Intersection_set)/len(Union_set)) *100
    
    first_model_name =  extract_model_name(csv_files[0])
    second_model_name =  extract_model_name(csv_files[1])

    venn2(subsets = (len(scan_sets[0]),len(Intersection_set), len(scan_sets[1])), set_labels = (first_model_name, second_model_name))
    plt.savefig(FIGURE_NAME)
