from pathlib import Path
import sys

sys.path.append('./src/common/depthmap_toolkit')
from utils import parse_calibration  # noqa: E402

TOOLKIT_DIR = Path(__file__).parents[0].absolute()


def test_parse_calibration():
    calibration_file = TOOLKIT_DIR / 'huawei_p40pro/camera_calibration.txt'
    calibration = parse_calibration(calibration_file)
    expected = [
        [0.6786797, 0.90489584, 0.49585155, 0.5035042],
        [0.6786797, 0.90489584, 0.49585155, 0.5035042],
        [0., 0., 0.],
    ]
    assert calibration == expected
