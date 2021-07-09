import argparse
import logging
import logging.config
import os
import random
import shutil
from importlib import import_module
from pathlib import Path
from typing import List

from bunch import Bunch
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run

from constants import DATA_DIR_ONLINE_RUN, DEFAULT_CONFIG, REPO_DIR


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


def is_offline_run(run: Run) -> bool:
    return run.id.startswith("OfflineRun")


# Get the current run.
RUN = Run.get_context()

if is_offline_run(RUN):
    # Copy common into the temp folder
    common_dir_path = REPO_DIR / "src/common"
    temp_common_dir = Path(__file__).parent / "temp_common"
    copy_dir(src=common_dir_path, tgt=temp_common_dir, glob_pattern='*/*.py', should_touch_init=True)

from temp_common.evaluation.eval_utilities import (  # noqa: E402, F401
    Evaluation, EnsembleEvaluation, MultiartifactEvaluation,
    download_dataset, get_dataset_path)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_config_module", default=DEFAULT_CONFIG, help="Configuration file")
    args = parser.parse_args()
    qa_config_module = args.qa_config_module
    qa_config = import_module(qa_config_module)
else:
    qa_config_module = DEFAULT_CONFIG
    qa_config = import_module(qa_config_module)
logging.info('Using the following config: %s', qa_config_module)


MODEL_CONFIG = qa_config.MODEL_CONFIG
EVAL_CONFIG = qa_config.EVAL_CONFIG
DATA_CONFIG = qa_config.DATA_CONFIG
RESULT_CONFIG = qa_config.RESULT_CONFIG
FILTER_CONFIG = qa_config.FILTER_CONFIG if getattr(qa_config, 'FILTER_CONFIG', False) else None

RUN_ID = MODEL_CONFIG.RUN_ID if getattr(MODEL_CONFIG, 'RUN_ID', False) else None
RUN_IDS = MODEL_CONFIG.RUN_IDS if getattr(MODEL_CONFIG, 'RUN_IDS', False) else None
assert bool(RUN_ID) != bool(RUN_IDS), 'RUN_ID xor RUN_IDS needs to be defined'


class RunInitializer:
    """Run azure setup and prepare dataset"""
    def __init__(self, data_config: Bunch) -> None:
        self._data_config = data_config
        self.run_azureml_setup()
        self.get_dataset()

    def run_azureml_setup(self):
        raise NotImplementedError

    def get_dataset(self):
        raise NotImplementedError


class OfflineRunInitializer(RunInitializer):
    """Offline run. Download the sample dataset and run locally. Still push results to Azure"""
    def __init__(self, data_config: Bunch) -> None:
        super().__init__(data_config)

    def run_azureml_setup(self):
        logging.info("Running in offline mode...")

        logging.info("Accessing workspace...")
        self.workspace = Workspace.from_config()
        self.experiment = Experiment(self.workspace, EVAL_CONFIG.EXPERIMENT_NAME)
        self.run = self.experiment.start_logging(outputs=None, snapshot_directory=None)

    def get_dataset(self):
        logging.info("Accessing dataset...")
        dataset_name = self._data_config.NAME
        self.dataset_path = str(REPO_DIR / "data" / dataset_name)
        if not os.path.exists(self.dataset_path):
            dataset = self.workspace.datasets[dataset_name]
            dataset.download(target_path=self.dataset_path, overwrite=False)


class OnlineRunInitializer(RunInitializer):
    def __init__(self, data_config: Bunch, run: Run) -> None:
        self.run = run
        super().__init__(data_config)

    def run_azureml_setup(self):
        logging.info("Running in online mode...")
        self.experiment = self.run.experiment
        self.workspace = self.experiment.workspace

    def get_dataset(self):
        dataset_name = self._data_config.NAME
        # Download
        self.dataset_path = get_dataset_path(DATA_DIR_ONLINE_RUN, dataset_name)
        download_dataset(self.workspace, dataset_name, self.dataset_path)


def is_ensemble_evaluation(model_config: Bunch) -> bool:
    return getattr(model_config, 'RUN_IDS', False)


def is_multiartifact_evaluation(data_config: Bunch) -> bool:
    return getattr(data_config, "N_ARTIFACTS", 1) > 1


if __name__ == "__main__":

    # Make experiment reproducible
    tf.random.set_seed(EVAL_CONFIG.SPLIT_SEED)
    random.seed(EVAL_CONFIG.SPLIT_SEED)

    if is_offline_run(RUN):
        OUTPUT_CSV_PATH = str(REPO_DIR / 'data' / RESULT_CONFIG.SAVE_PATH)
        initializer = OfflineRunInitializer(DATA_CONFIG)
    else:
        OUTPUT_CSV_PATH = RESULT_CONFIG.SAVE_PATH
        initializer = OnlineRunInitializer(DATA_CONFIG, RUN)

    if is_ensemble_evaluation(MODEL_CONFIG):
        MODEL_BASE_DIR = (REPO_DIR / 'data' / MODEL_CONFIG.EXPERIMENT_NAME) if is_offline_run(RUN) else Path('.')
        evaluation = EnsembleEvaluation(MODEL_CONFIG, MODEL_BASE_DIR, initializer.dataset_path)
        evaluation.get_the_model_path(initializer.workspace)
        model_paths = evaluation.model_paths
    elif is_multiartifact_evaluation(DATA_CONFIG):
        MODEL_BASE_DIR = REPO_DIR / 'data' / RUN_ID if is_offline_run(RUN) else Path('.')
        evaluation = MultiartifactEvaluation(MODEL_CONFIG, MODEL_BASE_DIR, initializer.dataset_path)
        evaluation.get_the_model_path(initializer.workspace)
        model_path = evaluation.model_path
    else:
        MODEL_BASE_DIR = REPO_DIR / 'data' / RUN_ID if is_offline_run(RUN) else Path('.')
        evaluation = Evaluation(MODEL_CONFIG, MODEL_BASE_DIR, initializer.dataset_path)
        evaluation.get_the_model_path(initializer.workspace)
        model_path = evaluation.model_path

    # Get the QR-code paths
    qrcode_paths = evaluation.get_the_qr_code_path()
    if getattr(EVAL_CONFIG, 'DEBUG_RUN', False) and len(qrcode_paths) > EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN:
        qrcode_paths = qrcode_paths[:EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN]
        logging.info("Executing on %d qrcodes for FAST RUN", EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN)

    # Is this a multiartifact model?
    if is_multiartifact_evaluation(DATA_CONFIG):
        predictions: List[List[str]] = evaluation.get_prediction_(model_path, qrcode_paths, DATA_CONFIG)
        df = evaluation.prepare_dataframe(predictions)

    else:  # Single-artifact
        dataset_evaluation, new_paths_evaluation = evaluation.prepare_dataset(qrcode_paths, DATA_CONFIG, FILTER_CONFIG)

        if is_ensemble_evaluation(MODEL_CONFIG):
            prediction_list_one = evaluation.get_prediction_(model_paths, dataset_evaluation, DATA_CONFIG)
        else:
            prediction_list_one = evaluation.get_prediction_(model_path, dataset_evaluation, DATA_CONFIG)
        logging.info("Prediction made by model on the depthmaps...")
        logging.info(prediction_list_one)

        df = evaluation.prepare_dataframe(
            new_paths_evaluation, prediction_list_one, DATA_CONFIG, RESULT_CONFIG)
        # df has columns: ['qrcode', 'artifact', 'scantype', 'GT', 'predicted']

    descriptor = MODEL_CONFIG.RUN_ID if getattr(MODEL_CONFIG, 'RUN_ID', False) else MODEL_CONFIG.EXPERIMENT_NAME
    evaluation.evaluate(df, DATA_CONFIG, RESULT_CONFIG, EVAL_CONFIG, OUTPUT_CSV_PATH, descriptor)

    # Done.
    initializer.run.complete()
