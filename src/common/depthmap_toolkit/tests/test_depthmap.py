from pathlib import Path
import sys

sys.path.append('./src/common/depthmap_toolkit')
from depthmap import Depthmap  # noqa: E402

TOOLKIT_DIR = Path(__file__).parents[0].absolute()


def test_depthmap():
    depthmap_dir = str(TOOLKIT_DIR / 'huawei_p40pro')
    depthmap_fname = 'depth_dog_1622182020448_100_234.depth'
    rgb_fname = 'rgb_dog_1622182020448_100_234.jpg'

    depthmap = Depthmap.create_from_file(depthmap_dir, depthmap_fname, rgb_fname)

    assert depthmap.width == 240
    assert depthmap.height == 180


if __name__ == '__main__':
    test_depthmap()