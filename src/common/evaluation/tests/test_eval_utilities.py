import sys
from pathlib import Path

from bunch import Bunch

sys.path.append(str(Path(__file__).parents[2]))  # common/ dir

from evaluation.eval_utilities import (  # noqa: E402
    Evaluation)

MODEL_CONFIG = Bunch(dict(
        EXPERIMENT_NAME='q3-depthmap-plaincnn-height-95k',
        RUN_ID='q3-depthmap-plaincnn-height-95k_1617983539_763a24b9',  # Run 9
        INPUT_LOCATION='outputs',
        NAME='best_model.ckpt',
    ))
CWD = Path(__file__).parent


def test_evaluation_get_the_qr_code_path():
    model_base_dir = '/tmp/models/'
    evaluation = Evaluation(MODEL_CONFIG, model_base_dir)
    dataset_path = CWD / "test_data/anontest-depthmap-mini"
    qrcode_paths = evaluation.get_the_qr_code_path(dataset_path)
    assert len(qrcode_paths) == 2
    qrcode_paths = sorted(qrcode_paths)
    expected_path = CWD / 'test_data/anontest-depthmap-mini/scans/1583462470-16tvfmb1d0'
    assert Path(qrcode_paths[0]) == expected_path

if __name__ == "__main__":
    test_evaluation_get_the_qr_code_path()
