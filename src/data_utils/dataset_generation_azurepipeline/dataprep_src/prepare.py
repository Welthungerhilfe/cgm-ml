import os
from pathlib import Path
import sys

from azureml.core import Experiment, Workspace
from azureml.core.run import Run
import pandas as pd


def print_blob_files(path):
    print("CGM blob paths:")
    p = path.glob('*')  # path.glob('**/*')
    files = [x for x in p]
    print(f"Num files: {len(files)}")
    print(files[:3])


print("Arguments: ")
print(sys.argv)

run = Run.get_context()
logging.info('Running in online mode...')
experiment = run.experiment
workspace = experiment.workspace

# SQL dataset
tabular_dataset = run.input_datasets['input1']
df = tabular_dataset.to_pandas_dataframe()
print("CGM Dataframe:")
print(df.head())

# Blob dataset
blob_dataset_path = run.input_datasets['input2']
print_blob_files(Path(blob_dataset_path))
