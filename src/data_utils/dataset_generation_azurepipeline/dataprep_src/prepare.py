import logging
import os
from pathlib import Path
import sys

from azureml.core import Experiment, Workspace
from azureml.core.run import Run
import pandas as pd


REPO_DIR = Path(__file__).parents[4].absolute()
print(f"REPO_DIR: {REPO_DIR}")


def print_blob_files(path):
    print("CGM blob paths:")
    p = path.glob('*')  # path.glob('**/*')
    files = [x for x in p]
    print(f"Num files: {len(files)}")
    print(files[:3])


print("Arguments: ")
print(sys.argv)

# Setup
run = Run.get_context()
if run.id.startswith("OfflineRun"):
    workspace = Workspace.from_config()
    experiment = Experiment(workspace, "training-junkyard")
else:
    experiment = run.experiment
    workspace = experiment.workspace


# SQL dataset
if run.id.startswith("OfflineRun"):
    df = pd.read_csv(REPO_DIR / 'data' / 'sql_query_result.csv')
else:
    tabular_dataset = run.input_datasets['input1']
    df = tabular_dataset.to_pandas_dataframe()
print("CGM Dataframe:")
print(df.head())


# Blob dataset
if run.id.startswith("OfflineRun"):
    pass
else:
    blob_dataset_path = run.input_datasets['input2']
    print_blob_files(Path(blob_dataset_path))
