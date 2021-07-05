from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory

sys.path.append('./src/common/data_utilities')
from mlpipeline_utils import (  # noqa: E402
    load_depth, get_depthmaps, ArtifactProcessor)

DATA_UTILITIES_DIR = Path(__file__).parents[1].absolute()
ARTIFACT_ZIP_PATH = 'be1faf54-69c7-11eb-984b-a3ffd42e7b5a/depth/bd67cd9e-69c7-11eb-984b-77ac9d2b4986'
ARTIFACT_TUPLE = (
    ARTIFACT_ZIP_PATH,
    '2021-04-22_13-34-33-302557',
    'c571de02-a723-11eb-8845-bb6589a1fbe8',
    102,
    50.0,
    20.555,
    10.0,
    3,
)
_COLUMN_NAMES = ['file_path', 'timestamp', 'scan_id', 'scan_step', 'height', 'weight', 'muac', 'order_number']
IDX2COL = {i: col for i, col in enumerate(_COLUMN_NAMES)}


def test_load_depth_hugh():
    depth_file = DATA_UTILITIES_DIR / 'tests' / 'zip_files' / ARTIFACT_ZIP_PATH
    data, _width, _height, _depth_scale, _max_confidence = load_depth(depth_file)
    assert isinstance(data, bytes)


def test_artifact_processor():
    input_dir = DATA_UTILITIES_DIR / 'tests' / 'zip_files'

    with TemporaryDirectory() as output_dir:
        artifact_processor = ArtifactProcessor(input_dir, output_dir, IDX2COL)
        processed_fname = artifact_processor.process_artifact_tuple(ARTIFACT_TUPLE)

        depthmap, targets = pickle.load(open(processed_fname, 'rb'))
        assert depthmap.shape == (180, 240, 1), depthmap.shape
        assert len(targets) == 3

        pickle_path_expected = str(
            DATA_UTILITIES_DIR
            / 'tests'
            / 'pickle_files'
            / 'scans'
            / 'c571de02-a723-11eb-8845-bb6589a1fbe8'
            / '102'
            / 'pc_c571de02-a723-11eb-8845-bb6589a1fbe8_2021-04-22_13-34-33-302557_102_3.p')
        assert pickle_path_expected.split('/')[-4:] == processed_fname.split('/')[-4:]


def test_get_depthmaps():
    zip_input_full_path = DATA_UTILITIES_DIR / 'tests' / 'zip_files' / ARTIFACT_ZIP_PATH
    preprocessed_depthmaps = get_depthmaps([zip_input_full_path])
    assert preprocessed_depthmaps[0].shape == (180, 240, 1), preprocessed_depthmaps[0].shape
