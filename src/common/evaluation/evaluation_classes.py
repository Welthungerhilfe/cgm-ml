import logging
import logging.config
import os
from pathlib import Path
import sys
import time
from typing import List, Tuple

import glob2 as glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from azureml.core import Workspace
from bunch import Bunch

from .constants_eval import (
    COLUMN_NAME_AGE, COLUMN_NAME_GOODBAD, COLUMN_NAME_SEX,
    GOODBAD_IDX, GOODBAD_DICT, SEX_IDX, AGE_IDX, HEIGHT_IDX, WEIGHT_IDX)
from .eval_utils import (
    avgerror, calculate_performance)
from .uncertainty_utils import get_prediction_uncertainty_deepensemble
from .eval_utilities import (
    download_model, get_model_path, get_depthmap_files, filter_dataset_according_to_standing_lying, tf_load_pickle,
    get_prediction, calculate_and_save_results, calculate_performance_age,
    draw_age_scatterplot, draw_stunting_diagnosis, draw_wasting_diagnosis,
    calculate_performance_sex, calculate_performance_goodbad, get_predictions_from_multiple_models,
    draw_uncertainty_goodbad_plot, get_prediction_multiartifact, get_column_list, prepare_sample_dataset,
    draw_uncertainty_scatterplot)

sys.path.append(str(Path(__file__).parents[1]))

from model_utils.preprocessing_multiartifact_python import create_multiartifact_paths_for_qrcodes  # noqa: E402
from model_utils.preprocessing_multiartifact_tensorflow import create_multiartifact_sample  # noqa: E402


class Evaluation:
    def __init__(self, model_config: Bunch, data_config: Bunch, model_base_dir: Path, dataset_path: str) -> None:
        self.model_config = model_config
        self.data_config = data_config
        self.model_base_dir = model_base_dir
        self.dataset_path = dataset_path
        self._input_location = os.path.join(self.model_config.INPUT_LOCATION, self.model_config.NAME)

    def get_the_model_path(self, workspace: Workspace):
        logging.info(f"Model will download from '{self._input_location}' to '{self.model_base_dir}'")
        download_model(workspace=workspace,
                       experiment_name=self.model_config.EXPERIMENT_NAME,
                       run_id=self.model_config.RUN_ID,
                       input_location=self._input_location,
                       output_location=self.model_base_dir)
        logging.info("Model was downloaded")
        self.model_path = self.model_base_dir / get_model_path(self.model_config)
        self.model_path_or_paths = self.model_path

    def get_the_qr_code_path(self) -> List[str]:
        dataset_path = os.path.join(self.dataset_path, "scans")
        logging.info('Dataset path: %s', dataset_path)
        logging.info('Getting QR-code paths...')
        qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
        logging.info('qrcode_paths: %d', len(qrcode_paths))
        assert len(qrcode_paths) != 0

        logging.info('Paths for evaluation: \n\t' + '\n\t'.join(qrcode_paths))
        logging.info(len(qrcode_paths))
        return qrcode_paths

    def prepare_dataset(self,
                        qrcode_paths: List[str],
                        filter_config: Bunch) -> Tuple[tf.data.Dataset, List[str]]:
        # Get depthmaps
        logging.info("Getting Depthmap paths...")
        paths_evaluation = get_depthmap_files(qrcode_paths)
        del qrcode_paths

        logging.info("Using %d artifact files for evaluation.", len(paths_evaluation))

        paths_belonging_to_predictions = paths_evaluation

        if filter_config is not None and filter_config.IS_ENABLED:
            standing = load_model(filter_config.NAME)
            paths_belonging_to_predictions = filter_dataset_according_to_standing_lying(paths_evaluation, standing)

        logging.info("Creating dataset for training.")
        paths = paths_belonging_to_predictions
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        dataset_norm = dataset.map(
            lambda path: tf_load_pickle(path, self.data_config.NORMALIZATION_VALUE, self.data_config)
        )

        # filter goodbad==delete
        if GOODBAD_IDX in self.data_config.TARGET_INDEXES:
            goodbad_index = self.data_config.TARGET_INDEXES.index(GOODBAD_IDX)
            dataset_norm = dataset_norm.filter(
                lambda _path, _depthmap, targets: targets[goodbad_index] != GOODBAD_DICT['delete'])

        dataset_norm = dataset_norm.cache()
        dataset_norm = dataset_norm.prefetch(tf.data.experimental.AUTOTUNE)
        temp_dataset_evaluation = dataset_norm
        del dataset_norm
        logging.info("Created dataset for training.")

        # Update paths_belonging_to_predictions after filtering
        dataset_paths = temp_dataset_evaluation.map(lambda path, _depthmap, _targets: path)
        list_paths = list(dataset_paths.as_numpy_iterator())
        paths_belonging_to_predictions = [x.decode() for x in list_paths]

        dataset_evaluation = temp_dataset_evaluation.map(lambda _path, depthmap, targets: (depthmap, targets))
        del temp_dataset_evaluation
        return dataset_evaluation, paths_belonging_to_predictions

    def get_prediction_(self, model_path: Path, dataset_evaluation: tf.data.Dataset) -> np.array:
        return get_prediction(model_path, dataset_evaluation, self.data_config)

    def prepare_dataframe(self,
                          paths_belonging_to_predictions: List[str],
                          prediction_list_one: np.array,
                          result_config: Bunch) -> pd.DataFrame:
        qrcode_list, scantype_list, artifact_list, prediction_list, target_list = get_column_list(
            paths_belonging_to_predictions, prediction_list_one, self.data_config)

        df = pd.DataFrame({
            'qrcode': qrcode_list,
            'artifact': artifact_list,
            'scantype': scantype_list,
            'GT': target_list if target_list[0].shape == tuple() else [el[0] for el in target_list],
            'predicted': prediction_list
        }, columns=result_config.COLUMNS)
        df['GT'] = df['GT'].astype('float64')
        df['predicted'] = df['predicted'].astype('float64')

        if 'AGE_BUCKETS' in result_config.keys():
            idx = self.data_config.TARGET_INDEXES.index(AGE_IDX)
            df[COLUMN_NAME_AGE] = [el[idx] for el in target_list]
        if SEX_IDX in self.data_config.TARGET_INDEXES:
            idx = self.data_config.TARGET_INDEXES.index(SEX_IDX)
            df[COLUMN_NAME_SEX] = [el[idx] for el in target_list]
        if GOODBAD_IDX in self.data_config.TARGET_INDEXES:
            idx = self.data_config.TARGET_INDEXES.index(GOODBAD_IDX)
            df[COLUMN_NAME_GOODBAD] = [el[idx] for el in target_list]

        logging.info("df.shape: %s", df.shape)
        return df

    def evaluate(self,
                 df: pd.DataFrame,
                 result_config: Bunch,
                 eval_config: Bunch,
                 OUTPUT_CSV_PATH: str,
                 descriptor: str):
        df_grouped = df.groupby(['qrcode', 'scantype']).mean()
        logging.info("Mean Avg Error: %s", df_grouped)

        df_grouped['error'] = df_grouped.apply(avgerror, axis=1)

        csv_fpath = f"{OUTPUT_CSV_PATH}/{descriptor}.csv"
        logging.info("Calculate and save the results to %s", csv_fpath)
        calculate_and_save_results(df_grouped, eval_config.NAME, csv_fpath,
                                   self.data_config, result_config, fct=calculate_performance)

        sample_csv_fpath = f"{OUTPUT_CSV_PATH}/inaccurate_scans_{descriptor}.csv"
        df_grouped.to_csv(sample_csv_fpath, index=True)

        if 'AGE_BUCKETS' in result_config.keys():
            csv_fpath = f"{OUTPUT_CSV_PATH}/age_evaluation_{descriptor}.csv"
            logging.info("Calculate and save age results to %s", csv_fpath)
            calculate_and_save_results(df_grouped, eval_config.NAME, csv_fpath,
                                       self.data_config, result_config, fct=calculate_performance_age)
            png_fpath = f"{OUTPUT_CSV_PATH}/age_evaluation_scatter_{descriptor}.png"
            logging.info("Calculate and save scatterplot results to %s", png_fpath)
            draw_age_scatterplot(df, png_fpath)

        if (HEIGHT_IDX in self.data_config.TARGET_INDEXES
                and AGE_IDX in self.data_config.TARGET_INDEXES
                and descriptor != self.model_config.EXPERIMENT_NAME):
            png_fpath = f"{OUTPUT_CSV_PATH}/stunting_diagnosis_{descriptor}.png"
            logging.info("Calculate zscores and save confusion matrix results to %s", png_fpath)
            start = time.time()
            draw_stunting_diagnosis(df, png_fpath)
            end = time.time()
            logging.info("Total time for Calculate zscores and save confusion matrix: %.2f", end - start)

        if (WEIGHT_IDX in self.data_config.TARGET_INDEXES
                and AGE_IDX in self.data_config.TARGET_INDEXES
                and descriptor != self.model_config.EXPERIMENT_NAME):
            png_fpath = f"{OUTPUT_CSV_PATH}/wasting_diagnosis_{descriptor}.png"
            logging.info("Calculate and save wasting confusion matrix results to %s", png_fpath)
            start = time.time()
            draw_wasting_diagnosis(df, png_fpath)
            end = time.time()
            logging.info("Total time for Calculate zscores and save wasting confusion matrix: %.2f", end - start)

        if SEX_IDX in self.data_config.TARGET_INDEXES:
            csv_fpath = f"{OUTPUT_CSV_PATH}/sex_evaluation_{descriptor}.csv"
            logging.info("Calculate and save sex results to %s", csv_fpath)
            calculate_and_save_results(df_grouped, eval_config.NAME, csv_fpath,
                                       self.data_config, result_config, fct=calculate_performance_sex)
        if GOODBAD_IDX in self.data_config.TARGET_INDEXES:
            csv_fpath = f"{OUTPUT_CSV_PATH}/goodbad_evaluation_{descriptor}.csv"
            logging.info("Calculate performance on bad/good scans and save results to %s", csv_fpath)
            calculate_and_save_results(df_grouped, eval_config.NAME, csv_fpath,
                                       self.data_config, result_config, fct=calculate_performance_goodbad)


class EnsembleEvaluation(Evaluation):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_the_model_path(self, workspace: Workspace):
        """Get multiple model paths"""
        for run_id in self.model_config.RUN_IDS:
            logging.info(f"Downloading run {run_id}")
            download_model(
                workspace=workspace,
                experiment_name=self.model_config.EXPERIMENT_NAME,
                run_id=run_id,
                input_location=self._input_location,
                output_location=self.model_base_dir / run_id
            )
        model_paths = glob.glob(os.path.join(self.model_base_dir, "*"))
        model_paths = [p for p in model_paths if os.path.isdir(p)]
        model_paths = [p for p in model_paths if p.split("/")[-1].startswith(self.model_config.EXPERIMENT_NAME)]
        model_paths = [os.path.join(p, self.model_config.INPUT_LOCATION, self.model_config.NAME) for p in model_paths]
        logging.info(f"Models paths ({len(model_paths)}):")
        logging.info("\t" + "\n\t".join(model_paths))
        self.model_paths = model_paths
        self.model_path_or_paths = model_paths

    def get_prediction_(self,
                        model_paths: List[Path],
                        dataset_evaluation: tf.data.Dataset) -> np.array:
        return get_predictions_from_multiple_models(model_paths, dataset_evaluation, self.data_config)

    def evaluate(self,
                 df: pd.DataFrame,
                 result_config: Bunch,
                 eval_config: Bunch,
                 OUTPUT_CSV_PATH: str,
                 descriptor: str):
        super().evaluate(df, result_config, eval_config, OUTPUT_CSV_PATH, descriptor)

        if not result_config.USE_UNCERTAINTY:
            return

        assert GOODBAD_IDX in self.data_config.TARGET_INDEXES
        assert COLUMN_NAME_GOODBAD in df

        # Sample one artifact per scan (qrcode, scantype combination)
        df_sample = df.groupby(['qrcode', 'scantype']).apply(lambda x: x.sample(1))

        # Prepare uncertainty prediction on these artifacts
        dataset_sample = prepare_sample_dataset(df_sample, self.dataset_path, self.data_config)

        # Predict uncertainty
        uncertainties = get_prediction_uncertainty_deepensemble(self.model_paths, dataset_sample)

        assert len(df_sample) == len(uncertainties)
        df_sample['uncertainties'] = uncertainties

        png_fpath = f"{OUTPUT_CSV_PATH}/uncertainty_distribution.png"
        draw_uncertainty_goodbad_plot(df_sample, png_fpath)

        df_sample_100 = df_sample.iloc[df_sample.index.get_level_values('scantype') == '100']
        png_fpath = f"{OUTPUT_CSV_PATH}/uncertainty_code100_distribution.png"
        draw_uncertainty_goodbad_plot(df_sample_100, png_fpath)

        png_fpath = f"{OUTPUT_CSV_PATH}/uncertainty_scatter_distribution.png"
        draw_uncertainty_scatterplot(df_sample, png_fpath)

        # Filter for scans with high certainty and calculate their accuracy/results
        df_sample['error'] = df_sample.apply(avgerror, axis=1).abs()
        df_sample_better_threshold = df_sample[df_sample['uncertainties'] < result_config.UNCERTAINTY_THRESHOLD_IN_CM]
        csv_fpath = f"{OUTPUT_CSV_PATH}/uncertainty_smaller_than_{result_config.UNCERTAINTY_THRESHOLD_IN_CM}cm.csv"
        logging.info("Uncertainty: For more certain than %.2f cm, calculate and save the results to %s",
                     result_config.UNCERTAINTY_THRESHOLD_IN_CM, csv_fpath)
        calculate_and_save_results(df_sample_better_threshold, eval_config.NAME, csv_fpath,
                                   self.data_config, result_config, fct=calculate_performance)


class MultiartifactEvaluation(Evaluation):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def prepare_dataset(self,
                        qrcode_paths: List[str],
                        filter_config: Bunch) -> Tuple[tf.data.Dataset, List[List[str]]]:
        samples_paths = create_multiartifact_paths_for_qrcodes(qrcode_paths, self.data_config)

        depthmaps, targets = [], []
        for sample_paths in samples_paths:
            depthmap, target = create_multiartifact_sample(sample_paths,
                                                           self.data_config.NORMALIZATION_VALUE,
                                                           self.data_config.IMAGE_TARGET_HEIGHT,
                                                           self.data_config.IMAGE_TARGET_WIDTH,
                                                           tf.constant(self.data_config.TARGET_INDEXES),
                                                           self.data_config.N_ARTIFACTS)
            depthmaps.append(depthmap)
            targets.append(target)

        dataset = tf.data.Dataset.from_tensor_slices((depthmaps, targets))
        dataset = dataset.batch(self.data_config.BATCH_SIZE)

        return dataset, samples_paths

    def get_prediction_(self,
                        model_path: Path,
                        dataset_evaluation: List[str]) -> np.array:
        predictions = get_prediction_multiartifact(model_path, dataset_evaluation, self.data_config)
        return predictions

    def prepare_dataframe(self,
                          paths_belonging_to_predictions: List[List[str]],
                          prediction_list_one: np.array,
                          result_config: Bunch):
        first_paths = [paths_list[0] for paths_list in paths_belonging_to_predictions]
        qrcode_list, scantype_list, artifact_list, prediction_list, target_list = get_column_list(
            first_paths, prediction_list_one, self.data_config)

        df = pd.DataFrame({
            'qrcode': qrcode_list,
            'artifact': artifact_list,
            'scantype': scantype_list,
            'GT': target_list if target_list[0].shape == tuple() else [el[0] for el in target_list],
            'predicted': prediction_list
        }, columns=result_config.COLUMNS)
        df['GT'] = df['GT'].astype('float64')
        df['predicted'] = df['predicted'].astype('float64')

        if 'AGE_BUCKETS' in result_config.keys():
            idx = self.data_config.TARGET_INDEXES.index(AGE_IDX)
            df[COLUMN_NAME_AGE] = [el[idx] for el in target_list]
        if SEX_IDX in self.data_config.TARGET_INDEXES:
            idx = self.data_config.TARGET_INDEXES.index(SEX_IDX)
            df[COLUMN_NAME_SEX] = [el[idx] for el in target_list]
        if GOODBAD_IDX in self.data_config.TARGET_INDEXES:
            idx = self.data_config.TARGET_INDEXES.index(GOODBAD_IDX)
            df[COLUMN_NAME_GOODBAD] = [el[idx] for el in target_list]

        logging.info("df.shape: %s", df.shape)
        return df
