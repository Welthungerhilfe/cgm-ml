from pathlib import Path
import sys

sys.path.append('./src/common/depthmap_toolkit') # noqa
import utils # noqa


REPO_DIR = str(Path(__file__).parents[0].absolute())
WIDTH = 180
HEIGHT = 135


def test_extract_depthmap():
    # Setup
    extracted_depth_file = REPO_DIR + '/static_files/data'

    # Exercise
    data, width, height, depth_scale, max_confidence, matrix = utils.parse_data(extracted_depth_file)

    # Verify
    assert (width, height) == (WIDTH, HEIGHT)

    # Cleanup - none required
