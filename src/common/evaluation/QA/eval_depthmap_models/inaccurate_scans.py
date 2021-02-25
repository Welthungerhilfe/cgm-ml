import argparse
import pandas as pd
import logging
from typing import List

from glob2 import glob
from matplotlib_venn import venn2   
from matplotlib import pyplot as plt



ACC = 2
CSV_PATH = f"./outputs/**/*_scans.csv"

def merge_qrc(row):
    scans = str(row['qrcode'])+'_'+str(row['scantype'])
    return scans

def filter_scans(dataframe: pd.DataFrame,accuracy: int):
    error = dataframe[(dataframe['error'] >= accuracy) | (dataframe['error'] <= -accuracy)]
    return error

def frame_to_set(dataframe: pd.DataFrame):
    return set(dataframe['name'].to_list())

def calculate_union(set1:set,set2:set):
    union_set = set1.union(set2)
    return union_set

def calculate_intersection(set1:set,set2:set):
    intersection_set = set1.intersection(set2)
    return intersection_set
    

def inaccurate_scans(file: List[str]):
    """Function to combine the models resultant csv files into a single file
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
    if len(csv_files) <= 0:
        logging.warning("No csv files found in output directory to combine.")
    scan_sets = []
    for file in csv_files:
        print('files:',file)
        frame_set = inaccurate_scans(file)
        scan_sets.append(frame_set)
    
    Union_set = calculate_union(scan_sets[0],scan_sets[1])
    Intersection_set = calculate_intersection(scan_sets[0],scan_sets[1])
    percentage  = (len(Intersection_set)/len(Union_set)) *100
    print("overlap_percentage:",percentage)

venn2(subsets = (len(scan_sets[0]),len(Intersection_set), len(scan_sets[1])), set_labels = ('dropout', 'no_dropout'))
plt.savefig('venn.png')
