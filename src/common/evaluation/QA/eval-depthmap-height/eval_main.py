import os
import time
import glob
from pathlib import Path
import shutil

from azureml.core import Workspace, Experiment, Run
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import MpiConfiguration
from azureml.train.dnn import TensorFlow
import pandas as pd

from auth import get_auth
from src.qa_config import MODEL_CONFIG, EVAL_CONFIG, DATA_CONFIG, RESULT_CONFIG
from src.utils import download_model

CWD = Path(__file__).parent
TAGS = {}


if __name__ == "__main__":

    # Create a temp folder
    code_dir = CWD / "src"
    paths = glob.glob(os.path.join(code_dir, "*.py"))
    print("paths:", paths)
    print("Creating temp folder...")
    temp_path = CWD / "tmp_eval"
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.mkdir(temp_path)
    for p in paths:
        shutil.copy(p, temp_path)
    print("Done.")

    auth = None if Run.get_context().id.startswith("OfflineRun") else get_auth()
    print("auth:", auth)
    ws = Workspace.from_config(auth=auth)

    # Copy model to temp folder
    download_model(ws=ws,
                experiment_name=MODEL_CONFIG.EXPERIMENT_NAME,
                run_id=MODEL_CONFIG.RUN_ID,
                input_location=os.path.join(MODEL_CONFIG.INPUT_LOCATION, MODEL_CONFIG.NAME),
                output_location=temp_path)

    experiment = Experiment(workspace=ws, name=EVAL_CONFIG.EXPERIMENT_NAME)

    # Find/create a compute target.
    try:
        # Compute cluster exists. Just connect to it.
        compute_target = ComputeTarget(workspace=ws, name=EVAL_CONFIG.CLUSTER_NAME)
        print("Found existing compute target.")
    except ComputeTargetException:
        print("Creating a new compute target...")
        compute_config = AmlCompute.provisioning_configuration(vm_size = 'Standard_NC6', max_nodes = 4)
        compute_target = ComputeTarget.create(ws, EVAL_CONFIG.CLUSTER_NAME, compute_config)
        compute_target.wait_for_completion(show_output = True, min_node_count = None, timeout_in_minutes = 20)
    print("Compute target:", compute_target)

    dataset = ws.datasets[DATA_CONFIG.NAME]
    print("dataset:", dataset)
    print("TF supported versions:", TensorFlow.get_supported_versions())

    #parameters used in the evaluation
    script_params = {f"--MODEL_{k}": v for k, v in MODEL_CONFIG.items()}
    script_params.update({f"--EVAL_{k}": v for k, v in EVAL_CONFIG.items()})
    script_params.update({f"--DATA_{k}": v for k, v in DATA_CONFIG.items()})
    script_params.update({f"--RESULT_{k}": v for k, v in RESULT_CONFIG.items()})
    print("script_params:", script_params)

    start = time.time()

    # Specify pip packages here.
    pip_packages = [
        "azureml-dataprep[fuse,pandas]",
        "glob2",
        "opencv-python==4.1.2.30",
        "matplotlib",
        "tensorflow-addons==0.11.2",
    ]

    # Create the estimator.
    estimator = TensorFlow(
        source_directory=temp_path,
        compute_target=compute_target,
        entry_script="evaluate.py",
        use_gpu=True,
        framework_version="2.2",
        inputs=[dataset.as_named_input("dataset").as_mount()],
        pip_packages=pip_packages,
        script_params=script_params
    )

    # Set compute target.
    estimator.run_config.target = compute_target

    # Run the experiment.
    run = experiment.submit(estimator, tags=TAGS)

    # Show run.
    print("Run:", run)

    #Check the logs of the current run until is complete
    run.wait_for_completion(show_output=True)

    #Print Completed when run is completed
    print("Run status:", run.get_status())

    end = time.time()
    print("Total time for evaluation experiment: {} sec".format(end-start))

    #Download the evaluation results of the model
    GET_CSV_FROM_EXPERIMENT_PATH = '.'
    run.download_file(RESULT_CONFIG.SAVE_PATH, GET_CSV_FROM_EXPERIMENT_PATH)
    print("Downloaded the result.csv")

    result = pd.read_csv('result.csv')
    print("Result:", result)

    #Delete temp folder
    shutil.rmtree(temp_path)
