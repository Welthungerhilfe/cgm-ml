import pandas as pd
import json
from pathlib import Path
import os
import sys

REPO_DIR = Path(__file__).parents[1].absolute()

# To run the toolkit
# python3 ConvertableData/data_convestion_toolkit.py wfh_girls_p_exp.txt
# wfh_girls_z_exp.txt wfh_girls_2_5_zscores.json Height

# Percentile data file name
percentile_data_path = sys.argv[1]

# SD data file name
sd_data_path = sys.argv[2]

# Output Data filename
output_file_name = sys.argv[3]

# Day if table is wfa,lhfa
# Height if table is wfh
# Lenght if table is wfl
zscore_type = sys.argv[4]


def convertTxtToJSON(percentile_data_path, sd_data_path, output_file_name, zscore_type):
    lsm_sd_data = []

    percentile_data_path = REPO_DIR / 'ConvertableData' / percentile_data_path
    sd_data_path = REPO_DIR / 'ConvertableData' / sd_data_path
    output_file_name = REPO_DIR / 'tables' / output_file_name

    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    df = pd.read_csv(percentile_data_path, sep='\t')
    df2 = pd.read_csv(sd_data_path, sep='\t')
    df = df.drop(['P01', 'P1', 'P3', 'P5', 'P10', 'P15', 'P25', 'P50',
                  'P75', 'P85', 'P90', 'P95', 'P97', 'P99', 'P999'], axis=1)
    df2 = df2.drop([zscore_type, 'SD4neg', 'SD4'], axis=1)
    result = pd.concat([df, df2], axis=1)
    for i in range(len(result)):
        dict = {}
        for j in (result.columns):
            if j == 'Day':
                g = int(result[j][i])
            else:
                g = float("{0:.5f}". format(result[j][i]))
            dict[j] = str(g)
        lsm_sd_data.append(dict)
    with open(output_file_name, 'w') as outfile:
        json.dump(lsm_sd_data, outfile, indent=4)


if __name__ == "__main__":
    convertTxtToJSON(percentile_data_path, sd_data_path,
                     output_file_name, zscore_type)
