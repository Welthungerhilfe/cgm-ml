from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys

from azureml.core import Experiment, Workspace
from azureml.core.run import Run
import pandas as pd

from mlpipeline_utils import ArtifactProcessor

DATASET_NAME = 'dataset'


def download_dataset(workspace: Workspace, dataset_name: str, dataset_path: str):
    logging.info("Accessing dataset...")
    if os.path.exists(dataset_path):
        return
    dataset = workspace.datasets[dataset_name]
    logging.info("Downloading dataset %s", dataset_name)
    dataset.download(target_path=dataset_path, overwrite=False)
    logging.info("Finished downloading %s", dataset_name)

def get_dataset_path(data_dir: Path, dataset_name: str) -> str:
    return str(data_dir / dataset_name)



def print_blob_files(path):
    print("CGM blob paths:")
    p = path.glob('*')  # path.glob('**/*')
    files = [x for x in p]
    print(f"Num files: {len(files)}")
    print(files[:3])


def parse_output_arg(argv):
    argv.index('--output')
    idx = argv.index('--output')
    return argv[idx+1]


if __name__ == '__main__':
    print("Arguments: ")
    print(sys.argv)

    # Setup
    run = Run.get_context()

    if run.id.startswith("OfflineRun"):
        REPO_DIR = Path(__file__).parents[4].absolute()

    if run.id.startswith("OfflineRun"):
        workspace = Workspace.from_config(Path(__file__).parents[1] / 'config.py')
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
        dataset_name = 'cgm-result-dataset'
        blob_dataset_path = get_dataset_path(REPO_DIR / 'data', dataset_name)
        download_dataset(workspace, dataset_name, blob_dataset_path)
    else:
        blob_dataset_path = run.input_datasets['input2']
        print_blob_files(Path(blob_dataset_path))


    # Output dataset
    if run.id.startswith("OfflineRun"):
        dataset_out_dir = datetime.now(timezone.utc).strftime(f"{DATASET_NAME}-%Y-%m-%d-%H-%M-%S")
        output_dir = REPO_DIR / 'data' / "cgm-datasets" / dataset_out_dir
    else:
        output_dir = parse_output_arg(sys.argv)
        print("output_dir: ")
        print(output_dir)

    # Transform
    artifact_processor = ArtifactProcessor(blob_dataset_path, output_dir, is_offline_run=run.id.startswith("OfflineRun"))
    for query_result in df.itertuples(index=False):
        res = artifact_processor.process_artifact_tuple(query_result)
        print(f"res{str(res)}")
