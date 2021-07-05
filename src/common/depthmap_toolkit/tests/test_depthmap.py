from pathlib import Path
import sys

sys.path.append('./src/common/depthmap_toolkit')
from depthmap import Depthmap  # noqa: E402

TOOLKIT_DIR = Path(__file__).parents[0].absolute()


def test_depthmap():
    depthmap_dir = str(TOOLKIT_DIR / 'huawei_p40pro')
    depthmap_fname = 'depth_dog_1622182020448_100_282.depth'
    rgb_fname = 'rgb_dog_1622182020448_100_282.jpg'
    calibration_file = str(TOOLKIT_DIR / 'huawei_p40pro' / 'camera_calibration.txt')

    dmap = Depthmap.create_from_file(depthmap_dir, depthmap_fname, rgb_fname, calibration_file)

    assert dmap.width == 240
    assert dmap.height == 180

    expected_intrinsics = [
        [0.6786797, 0.90489584, 0.49585155, 0.5035042],
        [0.6786797, 0.90489584, 0.49585155, 0.5035042],
    ]
    assert dmap.intrinsics == expected_intrinsics
    assert dmap.max_confidence == 7.
    assert dmap.depth_scale == 0.001

    floor = dmap.get_floor_level()
    mask = dmap.detect_child(floor)
    highest = dmap.get_highest_point(mask)
    child_height_in_m = highest[1] - floor
    assert 0 < child_height_in_m < 1.2
    assert mask.shape[0] == dmap.rgb_array.shape[1]
    assert mask.shape[1] == dmap.rgb_array.shape[0]

    angle_in_degrees = dmap.get_angle_between_camera_and_floor()
    assert -90 < angle_in_degrees < 90


def test_get_highest_point():
    depthmap_dir = str(TOOLKIT_DIR / 'huawei_p40pro')
    depthmap_fname = 'depth_dog_1622182020448_100_282.depth'
    rgb_fname = 'rgb_dog_1622182020448_100_282.jpg'
    calibration_file = str(TOOLKIT_DIR / 'huawei_p40pro' / 'camera_calibration.txt')
    dmap = Depthmap.create_from_file(depthmap_dir, depthmap_fname, rgb_fname, calibration_file)

    # Find top of the object
    floor = dmap.get_floor_level()
    mask = dmap.detect_child(floor)
    highest = dmap.get_highest_point(mask)  # 3D

    object_height_in_m = highest[1] - floor
    assert 0.3 < object_height_in_m < 0.6


if __name__ == '__main__':
    test_get_highest_point()
