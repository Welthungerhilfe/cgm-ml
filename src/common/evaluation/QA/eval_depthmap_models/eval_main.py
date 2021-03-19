import argparse
import glob
import os
import logging
import logging.config
import shutil
import time
from importlib import import_module
from pathlib import Path

import azureml._restclient.snapshots_client
from azureml.core import Experiment, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.run import Run
from azureml.train.dnn import TensorFlow

from src.constants import REPO_DIR, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


CWD = Path(__file__).parent
TAGS = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_config_module", default=DEFAULT_CONFIG, help="Configuration file")
    args = parser.parse_args()

    qa_config = import_module(f'src.{args.qa_config_module}')
    MODEL_CONFIG = qa_config.MODEL_CONFIG
    EVAL_CONFIG = qa_config.EVAL_CONFIG
    DATA_CONFIG = qa_config.DATA_CONFIG
    RESULT_CONFIG = qa_config.RESULT_CONFIG
    FILTER_CONFIG = qa_config.FILTER_CONFIG if getattr(qa_config, 'FILTER_CONFIG', False) else None

    # Create a temp folder
    code_dir = CWD / "src"
    paths = glob.glob(os.path.join(code_dir, "*.py"))
    logging.info("paths: %s", paths)
    logging.info("Creating temp folder...")
    temp_path = CWD / "tmp_eval"
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.mkdir(temp_path)
    for p in paths:
        shutil.copy(p, temp_path)
    logging.info("Done.")

    common_dir_path = REPO_DIR / "src/common"
    utils_paths = list(map(Path, glob.glob(os.path.join(common_dir_path, "*/*.py"))))
    temp_common_dir = Path(__file__).parent / "src/tmp_common"
    # Remove old temp_path
    if os.path.exists(temp_common_dir):
        shutil.rmtree(temp_common_dir)
    # Copy
    os.mkdir(temp_common_dir)
    os.system(f'touch {temp_common_dir}/__init__.py')
    for p in utils_paths:
        print(p)
        destpath = temp_common_dir / p.relative_to(common_dir_path)
        destpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, destpath)

    from src.tmp_common.evaluation.eval_utilities import download_model  # noqa: E402, F401

    ws = Workspace.from_config()

    run = Run.get_context()

    # When we run scripts locally(e.g. for debugging), we want to use another directory
    USE_LOCAL = False

    MODEL_BASE_DIR = REPO_DIR / 'data' / MODEL_CONFIG.RUN_ID if USE_LOCAL else temp_path
    logging.info('MODEL_BASE_DIR: %s', MODEL_BASE_DIR)
    os.makedirs(MODEL_BASE_DIR, exist_ok=True)

    # Copy model to temp folder
    download_model(ws=ws,
                   experiment_name=MODEL_CONFIG.EXPERIMENT_NAME,
                   run_id=MODEL_CONFIG.RUN_ID,
                   input_location=os.path.join(MODEL_CONFIG.INPUT_LOCATION, MODEL_CONFIG.NAME),
                   output_location=MODEL_BASE_DIR)

    # Copy filter to temp folder
    if FILTER_CONFIG is not None and FILTER_CONFIG.IS_ENABLED:
        download_model(ws=ws, experiment_name=FILTER_CONFIG.EXPERIMENT_NAME, run_id=FILTER_CONFIG.RUN_ID, input_location=os.path.join(
            FILTER_CONFIG.INPUT_LOCATION, MODEL_CONFIG.NAME), output_location=str(temp_path / FILTER_CONFIG.NAME))
        azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 500000000

    experiment = Experiment(workspace=ws, name=EVAL_CONFIG.EXPERIMENT_NAME)

    # Find/create a compute target.
    try:
        # Compute cluster exists. Just connect to it.
        compute_target = ComputeTarget(workspace=ws, name=EVAL_CONFIG.CLUSTER_NAME)
        logging.info("Found existing compute target.")
    except ComputeTargetException:
        logging.info("Creating a new compute target...")
        compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC6', max_nodes=4)
        compute_target = ComputeTarget.create(ws, EVAL_CONFIG.CLUSTER_NAME, compute_config)
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    logging.info("Compute target: %s", compute_target)

    dataset = ws.datasets[DATA_CONFIG.NAME]
    logging.info("dataset: %s", dataset)
    logging.info("TF supported versions: %s", TensorFlow.get_supported_versions())

    # parameters used in the evaluation
    script_params = {"--qa_config_module": args.qa_config_module}
    logging.info("script_params: %s", script_params)

    start = time.time()

    # Specify pip packages here.
    pip_packages = [
        "azureml-dataprep[fuse,pandas]",
        "glob2",
        "opencv-python==4.1.2.30",
        "matplotlib",
        "tensorflow-addons==0.11.2",
        "bunch==1.0.1",
        "cgmzscore==2.0.3",
        "scikit-learn==0.22.2.post1"
    ]

    # Create the estimator.
    estimator = TensorFlow(
        source_directory=temp_path,
        compute_target=compute_target,
        entry_script="evaluate.py",
        use_gpu=True,
        framework_version="2.3",
        inputs=[dataset.as_named_input("dataset").as_mount()],
        pip_packages=pip_packages,
        script_params=script_params
    )

    # Set compute target.
    estimator.run_config.target = compute_target

    # Run the experiment.
    run = experiment.submit(estimator, tags=TAGS)

    # Show run.
    logging.info("Run: %s", run)

    # Check the logs of the current run until is complete
    run.wait_for_completion(show_output=True)

    # Print Completed when run is completed
    logging.info("Run status: %s", run.get_status())

    end = time.time()
    logging.info("Total time for evaluation experiment: %d sec", end - start)

    # Download the evaluation results of the model
    GET_CSV_FROM_EXPERIMENT_PATH = '.'
    run.download_files(RESULT_CONFIG.SAVE_PATH, GET_CSV_FROM_EXPERIMENT_PATH)
    logging.info("Downloaded the result.csv")

    # Delete temp folder
    shutil.rmtree(temp_path)
