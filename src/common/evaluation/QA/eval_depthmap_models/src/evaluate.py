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
import numpy as np
import pandas as pd
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run
from tensorflow.keras.models import load_model

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


# Get the current run.
run = Run.get_context()

if run.id.startswith("OfflineRun"):

    # Copy common into the temp folder
    common_dir_path = REPO_DIR / "src/common"
    temp_common_dir = Path(__file__).parent / "temp_common"
    copy_dir(src=common_dir_path, tgt=temp_common_dir, glob_pattern='*/*.py', should_touch_init=True)

from temp_common.evaluation.eval_utils import (  # noqa: E402, F401
    avgerror, extract_qrcode, extract_scantype)
from temp_common.evaluation.eval_utilities import (  # noqa: E402, F401
    Evaluation, EnsembleEvaluation,
    download_dataset, get_dataset_path)
from temp_common.model_utils.preprocessing_multiartifact_python import \
    create_multiartifact_paths_for_qrcodes  # noqa: E402, F401
from temp_common.model_utils.preprocessing_multiartifact_tensorflow import \
    create_multiartifact_sample  # noqa: E402, F401


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


def get_prediction_multiartifact(model_path: str, samples_paths: List[List[str]]) -> List[List[str]]:
    """Make prediction on each multiartifact sample.

    Args:
        model_path: File path to the model
        samples_paths: A list of samples where each sample contains N_ARTIFACTS.

    Returns:
        List with tuples: ('artifacts', 'predicted', 'GT')
    """
    logging.info("loading model from %s", model_path)
    model = load_model(model_path, compile=False)

    predictions = []
    for sample_paths in samples_paths:
        depthmap, targets = create_multiartifact_sample(sample_paths,
                                                        DATA_CONFIG.NORMALIZATION_VALUE,
                                                        DATA_CONFIG.IMAGE_TARGET_HEIGHT,
                                                        DATA_CONFIG.IMAGE_TARGET_WIDTH,
                                                        tf.constant(DATA_CONFIG.TARGET_INDEXES),
                                                        DATA_CONFIG.N_ARTIFACTS)
        depthmaps = tf.stack([depthmap])
        pred = model.predict(depthmaps)
        predictions.append([sample_paths[0], float(np.squeeze(pred)), targets[0]])
    return predictions


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


if __name__ == "__main__":

    # Make experiment reproducible
    tf.random.set_seed(EVAL_CONFIG.SPLIT_SEED)
    random.seed(EVAL_CONFIG.SPLIT_SEED)

    OUTPUT_CSV_PATH = str(REPO_DIR / 'data'
                          / RESULT_CONFIG.SAVE_PATH) if run.id.startswith("OfflineRun") else RESULT_CONFIG.SAVE_PATH
    if RUN_ID is not None:
        MODEL_BASE_DIR = REPO_DIR / 'data' / RUN_ID if run.id.startswith("OfflineRun") else Path('.')
    if RUN_IDS is not None:
        MODEL_BASE_DIR = REPO_DIR / 'data' / \
            MODEL_CONFIG.EXPERIMENT_NAME if run.id.startswith("OfflineRun") else Path('.')

    if run.id.startswith("OfflineRun"):
        initializer = OfflineRunInitializer(DATA_CONFIG)
        run = initializer.run
    else:
        initializer = OnlineRunInitializer(DATA_CONFIG, run)

    workspace = initializer.workspace
    experiment = initializer.experiment
    run = initializer.run
    dataset_path = initializer.dataset_path

    input_location = os.path.join(MODEL_CONFIG.INPUT_LOCATION, MODEL_CONFIG.NAME)
    if RUN_IDS is not None:
        evaluation = EnsembleEvaluation(MODEL_CONFIG, MODEL_BASE_DIR, dataset_path)
        evaluation.get_the_model_path(workspace)
        model_paths = evaluation.model_paths
    else:
        evaluation = Evaluation(MODEL_CONFIG, MODEL_BASE_DIR, dataset_path)
        evaluation.get_the_model_path(workspace)
        model_path = evaluation.model_path

    # Get the QR-code paths
    qrcode_paths = evaluation.get_the_qr_code_path()
    if getattr(EVAL_CONFIG, 'DEBUG_RUN', False) and len(qrcode_paths) > EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN:
        qrcode_paths = qrcode_paths[:EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN]
        logging.info("Executing on %d qrcodes for FAST RUN", EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN)

    # Is this a multiartifact model?
    if getattr(DATA_CONFIG, "N_ARTIFACTS", 1) > 1:
        samples_paths = create_multiartifact_paths_for_qrcodes(qrcode_paths, DATA_CONFIG)
        predictions = get_prediction_multiartifact(model_path, samples_paths)

        df = pd.DataFrame(predictions, columns=['artifacts', 'predicted', 'GT'])
        df['scantype'] = df.apply(extract_scantype, axis=1)
        df['qrcode'] = df.apply(extract_qrcode, axis=1)
        MAE = df.groupby(['qrcode', 'scantype']).mean()
        MAE['error'] = MAE.apply(avgerror, axis=1)

    else:  # Single-artifact
        dataset_evaluation, new_paths_evaluation = evaluation.prepare_dataset(qrcode_paths, DATA_CONFIG, FILTER_CONFIG)

        if RUN_IDS is not None:
            prediction_list_one = evaluation.get_prediction_(model_paths, dataset_evaluation, DATA_CONFIG)
        if RUN_ID is not None:
            prediction_list_one = evaluation.get_prediction_(model_path, dataset_evaluation, DATA_CONFIG)
        logging.info("Prediction made by model on the depthmaps...")
        logging.info(prediction_list_one)

        df, target_list = evaluation.prepare_dataframe(new_paths_evaluation, prediction_list_one, DATA_CONFIG, FILTER_CONFIG, RESULT_CONFIG)

    evaluation.evaluate(df, target_list, DATA_CONFIG, RESULT_CONFIG, EVAL_CONFIG, OUTPUT_CSV_PATH)

    if RESULT_CONFIG.USE_UNCERTAINTY:
        evaluation.evaluate(df, DATA_CONFIG, RESULT_CONFIG, EVAL_CONFIG, FILTER_CONFIG, OUTPUT_CSV_PATH)

    # Done.
    run.complete()
