import pathlib
import sys

sys.path.append('./src/common/depthmap_toolkit')
import utils


CWD = pathlib.Path.cwd()
WIDTH = 240
HEIGHT = 180


def test_extract_depthmap():
    # Setup
    extracted_depth_file = CWD.joinpath('tests', 'static_files', 'data')

    # Exercise
    data, width, height, depth_scale, max_confidence, matrix = utils.parse_data(extracted_depth_file)

    # Verify
    assert (width, height) == (WIDTH, HEIGHT)

    # Cleanup - none required
