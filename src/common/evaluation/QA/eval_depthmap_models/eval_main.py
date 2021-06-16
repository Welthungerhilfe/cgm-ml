import argparse
import logging
import logging.config
import os
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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


CWD = Path(__file__).parent
TAGS = {}


def copy_dir(src: Path, tgt: Path, glob_pattern: str, should_touch_init: bool = False):
    logging.info("Creating temp folder")
    if tgt.exists():
        shutil.rmtree(tgt)
    tgt.mkdir(parents=True, exist_ok=True)
    if should_touch_init:
        (tgt / '__init__.py').touch(exist_ok=False)

    paths_to_copy = list(src.glob(glob_pattern))
    logging.info(f"Copying to {tgt} the following files: {str(paths_to_copy)}")
    for p in paths_to_copy:
        destpath = tgt / p.relative_to(src)
        destpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, destpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_config_module", default=DEFAULT_CONFIG, help="Configuration file")
    args = parser.parse_args()

    logging.info('Using the following config: %s', args.qa_config_module)
    qa_config = import_module(f'src.{args.qa_config_module}')
    MODEL_CONFIG = qa_config.MODEL_CONFIG
    EVAL_CONFIG = qa_config.EVAL_CONFIG
    DATA_CONFIG = qa_config.DATA_CONFIG
    RESULT_CONFIG = qa_config.RESULT_CONFIG
    FILTER_CONFIG = qa_config.FILTER_CONFIG if getattr(qa_config, 'FILTER_CONFIG', False) else None

    # Copy src/ dir
    temp_path = CWD / "temp_eval"
    copy_dir(src=CWD / "src", tgt=temp_path, glob_pattern='*.py')

    # Copy common into the temp folder
    common_dir_path = REPO_DIR / "src/common"
    temp_common_dir = temp_path / "temp_common"
    copy_dir(src=common_dir_path, tgt=temp_common_dir, glob_pattern='*/*.py', should_touch_init=True)

    from temp_eval.temp_common.evaluation.eval_utilities import download_model  # noqa: E402, F401

    workspace = Workspace.from_config()
    run = Run.get_context()

    # When we run scripts locally(e.g. for debugging), we want to use another directory
    USE_LOCAL = False

    RUN_ID = MODEL_CONFIG.RUN_ID if getattr(MODEL_CONFIG, 'RUN_ID', False) else None
    RUN_IDS = MODEL_CONFIG.RUN_IDS if getattr(MODEL_CONFIG, 'RUN_IDS', False) else None
    assert bool(RUN_ID) != bool(RUN_IDS), 'RUN_ID xor RUN_IDS needs to be defined'

    if RUN_ID:
        MODEL_BASE_DIR = REPO_DIR / 'data' / RUN_ID if USE_LOCAL else temp_path
    elif RUN_IDS:
        MODEL_BASE_DIR = REPO_DIR / 'data' / MODEL_CONFIG.EXPERIMENT_NAME if USE_LOCAL else temp_path
    logging.info('MODEL_BASE_DIR: %s', MODEL_BASE_DIR)
    MODEL_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Copy model to temp folder
    if RUN_ID:
        download_model(workspace=workspace,
                       experiment_name=MODEL_CONFIG.EXPERIMENT_NAME,
                       run_id=MODEL_CONFIG.RUN_ID,
                       input_location=os.path.join(MODEL_CONFIG.INPUT_LOCATION, MODEL_CONFIG.NAME),
                       output_location=MODEL_BASE_DIR)
    elif USE_LOCAL and RUN_IDS:
        for run_id in RUN_IDS:
            download_model(workspace=workspace,
                           experiment_name=MODEL_CONFIG.EXPERIMENT_NAME,
                           run_id=run_id,
                           input_location=os.path.join(MODEL_CONFIG.INPUT_LOCATION, MODEL_CONFIG.NAME),
                           output_location=MODEL_BASE_DIR / run_id)

    # Copy filter to temp folder
    if FILTER_CONFIG is not None and FILTER_CONFIG.IS_ENABLED:
        download_model(workspace,
                       experiment_name=FILTER_CONFIG.EXPERIMENT_NAME,
                       run_id=FILTER_CONFIG.RUN_ID,
                       input_location=os.path.join(FILTER_CONFIG.INPUT_LOCATION, MODEL_CONFIG.NAME),
                       output_location=str(temp_path / FILTER_CONFIG.NAME))
        azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 500000000

    experiment = Experiment(workspace=workspace, name=EVAL_CONFIG.EXPERIMENT_NAME)

    # Find/create a compute target.
    try:
        # Compute cluster exists. Just connect to it.
        compute_target = ComputeTarget(workspace=workspace, name=EVAL_CONFIG.CLUSTER_NAME)
        logging.info("Found existing compute target.")
    except ComputeTargetException:
        logging.info("Creating a new compute target...")
        compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC6', max_nodes=4)
        compute_target = ComputeTarget.create(workspace, EVAL_CONFIG.CLUSTER_NAME, compute_config)
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    logging.info("Compute target: %s", compute_target)

    dataset = workspace.datasets[DATA_CONFIG.NAME]
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
