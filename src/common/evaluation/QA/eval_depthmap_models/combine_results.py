import pandas as pd
from glob2 import glob
import os

if __name__ == "__main__":

    CSV_PATH  = '/common/evaluation/QA/eval_depthmap_models/*.csv'
    csv_files = glob(CSV_PATH)

    result_list =[]
    for results in csv_files:
        read_csv_file = pd.read_csv(results)
        result_list.append(read_csv_file)
        os.remove(results)

    final_result = pd.concat(result_list)
    final_result = final_result.round(2) 
    final_result.to_csv('results.csv',index=True)