from copy import copy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from bunch import Bunch
from tensorflow.keras import layers, models
import tensorflow as tf

sys.path.append(str(Path(__file__).parents[2]))  # common/ dir

from model_utils.model_plaincnn import create_cnn, create_base_cnn, create_head  # noqa: E402
from evaluation.evaluation_classes import Evaluation, EnsembleEvaluation, MultiartifactEvaluation  # noqa: E402

MODEL_CONFIG = Bunch(dict(
    EXPERIMENT_NAME='my-experiment',
    RUN_ID='my-experiment_1617983539_763a24b9',
    INPUT_LOCATION='outputs',
    NAME='best_model.ckpt',
))
EVAL_CONFIG = Bunch(dict(
    NAME='my-experiment-run_09',
    EXPERIMENT_NAME="QA-pipeline",
))
DATA_CONFIG = Bunch(dict(
    NAME='dataset-anontest',
    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,
    BATCH_SIZE=16,
    NORMALIZATION_VALUE=7.5,
    TARGET_INDEXES=[0],
    CODES=['100', '101', '102', '200', '201', '202'],
))
RESULT_CONFIG = Bunch(dict(
    ACCURACIES=[.2, .4, .6, 1., 1.2, 2., 2.5, 3., 4., 5., 6.],
    ACCURACY_MAIN_THRESH=1.0,
    COLUMNS=['qrcode', 'artifact', 'scantype', 'GT', 'predicted'],
    USE_UNCERTAINTY=False,
    SAVE_PATH='/tmp/config_test_eval_utilities',
))


CWD = Path(__file__).parent
DATASET_PATH = CWD / "test_data/anontest-depthmap-mini"

# Global constants used to test multiartifact model
N_ARTIFACTS = 2
IMAGE_TARGET_HEIGHT = 240
IMAGE_TARGET_WIDTH = 180
SAMPLING_STRATEGY_SYSTEMATIC = "systematic"


def prep_model(model_path: str):
    input_shape = (240, 180, 1)
    model = create_cnn(input_shape, dropout=False)
    model.save(model_path)


def prep_multiartifactlatefusion_model(model_path: str):
    # Create the base model
    input_shape = (240, 180, 1)
    base_model = create_base_cnn(input_shape, dropout=False)
    assert base_model.output_shape == (None, 128)

    # Create the head
    head_input_shape = (128 * N_ARTIFACTS, )
    head_model = create_head(head_input_shape, dropout=False)

    # Implement artifact flow through the same model
    model_input = layers.Input(shape=(IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH, N_ARTIFACTS))
    features_list = []
    for i in range(N_ARTIFACTS):
        features_part = model_input[:, :, :, i:i + 1]
        features_part = base_model(features_part)
        features_list.append(features_part)
    concatenation = layers.concatenate(features_list, axis=-1)
    assert concatenation.shape.as_list() == tf.TensorShape((None, 128 * N_ARTIFACTS)).as_list()
    model_output = head_model(concatenation)
    model = models.Model(model_input, model_output)
    model.save(model_path)


def test_evaluation_get_the_qr_code_path():
    evaluation = Evaluation(MODEL_CONFIG, DATA_CONFIG, model_base_dir=None, dataset_path=DATASET_PATH)
    qrcode_paths = evaluation.get_the_qr_code_path()
    assert len(qrcode_paths) == 2
    qrcode_paths = sorted(qrcode_paths)
    expected_path = CWD / 'test_data/anontest-depthmap-mini/scans/1583462470-16tvfmb1d0'
    assert Path(qrcode_paths[0]) == expected_path


def test_evaluation_prepare_dataset():
    # Prep
    evaluation = Evaluation(MODEL_CONFIG, DATA_CONFIG, model_base_dir=None, dataset_path=DATASET_PATH)
    qrcode_paths = evaluation.get_the_qr_code_path()
    # Run
    dataset, paths_belonging_to_predictions = evaluation.prepare_dataset(qrcode_paths, filter_config=None)
    # Test
    assert len(paths_belonging_to_predictions) == 5
    dataset_take = dataset.take(5)
    for sample in dataset_take:
        assert sample[0].shape == [240, 180, 1]
        assert sample[1].shape == [1]


def test_evaluation_evaluate():
    # Prep
    evaluation = Evaluation(MODEL_CONFIG, DATA_CONFIG, model_base_dir=None, dataset_path=DATASET_PATH)
    with TemporaryDirectory() as model_path:
        prep_model(model_path)
        qrcode_paths = evaluation.get_the_qr_code_path()
        dataset, paths_belonging_to_predictions = evaluation.prepare_dataset(qrcode_paths, filter_config=None)
        prediction_array = evaluation.get_prediction_(model_path, dataset)
        df = evaluation.prepare_dataframe(paths_belonging_to_predictions, prediction_array, RESULT_CONFIG)

    # Run
    with TemporaryDirectory() as output_csv_path:
        evaluation.evaluate(df, RESULT_CONFIG, EVAL_CONFIG, output_csv_path, descriptor=MODEL_CONFIG.RUN_ID)
        assert len(list(Path(output_csv_path).glob('*'))) == 2


def test_ensembleevaluation_evaluate():
    # Prep
    evaluation = EnsembleEvaluation(MODEL_CONFIG, DATA_CONFIG, model_base_dir=None, dataset_path=DATASET_PATH)
    with TemporaryDirectory() as models_path:
        model_paths = [Path(models_path) / 'model1', Path(models_path) / 'model2']
        for model_path in model_paths:
            prep_model(model_path)
        qrcode_paths = evaluation.get_the_qr_code_path()
        dataset, paths_belonging_to_predictions = evaluation.prepare_dataset(qrcode_paths, filter_config=None)
        prediction_array = evaluation.get_prediction_(model_paths, dataset)
        df = evaluation.prepare_dataframe(paths_belonging_to_predictions, prediction_array,
                                          RESULT_CONFIG)
    # Run
    with TemporaryDirectory() as output_csv_path:
        evaluation.evaluate(df, RESULT_CONFIG, EVAL_CONFIG, output_csv_path, descriptor=MODEL_CONFIG.EXPERIMENT_NAME)
        assert len(list(Path(output_csv_path).glob('*'))) == 2


def test_multiartifactevaluation_evaluate():
    data_config = copy(DATA_CONFIG)
    data_config.SAMPLING_STRATEGY = SAMPLING_STRATEGY_SYSTEMATIC
    data_config.N_ARTIFACTS = N_ARTIFACTS

    # Prep
    evaluation = MultiartifactEvaluation(MODEL_CONFIG, data_config, model_base_dir=None, dataset_path=DATASET_PATH)
    with TemporaryDirectory() as model_path:
        prep_multiartifactlatefusion_model(model_path)
        qrcode_paths = evaluation.get_the_qr_code_path()
        dataset, paths_belonging_to_predictions = evaluation.prepare_dataset(qrcode_paths, filter_config=None)
        prediction_array = evaluation.get_prediction_(model_path, dataset)
        df = evaluation.prepare_dataframe(paths_belonging_to_predictions, prediction_array, RESULT_CONFIG)

    # Run
    with TemporaryDirectory() as output_csv_path:
        evaluation.evaluate(df, RESULT_CONFIG, EVAL_CONFIG, output_csv_path, descriptor=MODEL_CONFIG.RUN_ID)
        assert len(list(Path(output_csv_path).glob('*'))) == 2
