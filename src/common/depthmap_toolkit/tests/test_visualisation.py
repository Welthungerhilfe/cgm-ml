from pathlib import Path
import sys

import numpy as np

sys.path.append('./src/common/depthmap_toolkit')
from depthmap import Depthmap  # noqa: E402
from visualisation import blur_face, render_pixel, SUBPLOT_RGB, SUBPLOT_COUNT  # noqa: E402

TOOLKIT_DIR = Path(__file__).parents[0].absolute()
OFFSET_X_Y = (0.37, 0.53)


def test_blur_face():
    depthmap_dir = str(TOOLKIT_DIR / 'huawei_p40pro')
    depthmap_fname = 'depth_dog_1622182020448_100_234.depth'
    rgb_fname = 'rgb_dog_1622182020448_100_234.jpg'
    calibration_file = str(TOOLKIT_DIR / 'huawei_p40pro' / 'camera_calibration.txt')
    dmap = Depthmap.create_from_file(depthmap_dir, depthmap_fname, rgb_fname, calibration_file)

    # Find top of the object
    floor = dmap.get_floor_level()
    mask = dmap.detect_child(floor)
    highest = dmap.get_highest_point(mask)  # 3D

    # Render the visualisations
    output_unblurred = np.zeros((dmap.width, dmap.height * SUBPLOT_COUNT, 3))
    for x in range(dmap.width):
        for y in range(dmap.height):
            render_pixel(output_unblurred, x, y, floor, mask, dmap)

    # Blur
    output_blurred = blur_face(output_unblurred, highest, dmap)

    # Assert some pixels in whole image change (image not same)
    all_count = dmap.width * dmap.height * SUBPLOT_COUNT
    count = np.count_nonzero(output_unblurred - output_blurred) / 3
    ratio_blurred = count / all_count
    assert 0.003 < ratio_blurred < 0.9

    # Assert that blurred around object
    offset = SUBPLOT_RGB * dmap.height
    object_x = int(dmap.width * OFFSET_X_Y[0])
    object_y = offset + int(dmap.height * (1.0 - OFFSET_X_Y[1]))
    slice_x = slice(object_x - 2, object_x + 2)
    slice_y = slice(object_y - 2, object_y + 2)
    assert (output_unblurred[slice_x, slice_y] != output_blurred[slice_x, slice_y]).any()

    # Assert that NOT blurred around corner
    corner_x = 0
    corner_y = offset
    slice_x = slice(corner_x, corner_x + 4)
    slice_y = slice(corner_y, corner_y + 4)
    assert (output_unblurred[slice_x, slice_y] == output_blurred[slice_x, slice_y]).all()
