import pandas as pd
from glob2 import glob
import logging

OUTPUT_FILE_NAME ='evaluated_models_result.csv'
CSV_PATH  = '/common/evaluation/QA/eval_depthmap_models/*.csv'

def combine_model_results(csv_file_list):
    """
    Function to combine the models resultant csv files into a single file

    Args:
        csv_file_list (list): list containing absolute path of csv file
    """
    if len(csv_file_list) <= 0:
        logging.warning("No csv files found in output directory to combine")
        return
    result_list = []
    for results in csv_files:
        read_csv_file = pd.read_csv(results,index_col=0)
        result_list.append(read_csv_file)
    final_result = pd.concat(result_list,axis=0)
    final_result = final_result.rename_axis("Model")
    final_result = final_result.round(2) 
    final_result.to_csv(OUTPUT_FILE_NAME,index=True)

if __name__ == "__main__":

    csv_files = glob(CSV_PATH)
    combine_model_results(csv_files)

    
        

 
    
    