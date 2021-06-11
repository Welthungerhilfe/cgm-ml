import logging
import logging.config

import matplotlib.pyplot as plt
import numpy as np

from constants import MASK_CHILD
from depthmap import Depthmap

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

PATTERN_LENGTH_IN_METERS = 0.1

SUBPLOT_DEPTH = 0
SUBPLOT_NORMAL = 1
SUBPLOT_SEGMENTATION = 2
SUBPLOT_CONFIDENCE = 3
SUBPLOT_RGB = 4
SUBPLOT_COUNT = 5


def render_pixel(output: object,
                 x: int,
                 y: int,
                 floor: float,
                 mask: np.array,
                 dmap: Depthmap):

    depth = dmap.parse_depth(x, y)
    if not depth:
        return

    # convert ToF coordinates into RGB coordinates
    vec = dmap.convert_2d_to_3d(1, x, y, depth)
    vec = dmap.convert_3d_to_2d(0, vec[0], vec[1], vec[2])

    # depth data visualisation (scaled to be visible)
    index = SUBPLOT_DEPTH * dmap.height + dmap.height - y - 1
    output[x][index] = 1.0 - min(depth / 2.0, 1.0)

    # normal vector visualisation
    normal = dmap.calculate_normal_vector(x, y)
    index = SUBPLOT_NORMAL * dmap.height + dmap.height - y - 1
    output[x][index][0] = abs(normal[0])
    output[x][index][1] = abs(normal[1])
    output[x][index][2] = abs(normal[2])

    # segmentation visualisation
    point = dmap.convert_2d_to_3d_oriented(1, x, y, depth)
    horizontal = (point[1] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
    vertical_x = (point[0] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
    vertical_z = (point[2] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
    vertical = (vertical_x + vertical_z) / 2.0
    index = SUBPLOT_SEGMENTATION * dmap.height + dmap.height - y - 1
    if mask[x][y] == MASK_CHILD:
        output[x][index][0] = horizontal / (depth * depth)
        output[x][index][1] = horizontal / (depth * depth)
    elif abs(normal[1]) < 0.5:
        output[x][index][0] = horizontal / (depth * depth)
    elif abs(normal[1]) > 0.5:
        if abs(point[1] - floor) < 0.1:
            output[x][index][2] = vertical / (depth * depth)
        else:
            output[x][index][1] = vertical / (depth * depth)

    # confidence value visualisation
    index = SUBPLOT_CONFIDENCE * dmap.height + dmap.height - y - 1
    output[x][index][:] = dmap.parse_confidence(x, y)
    if output[x][index][0] == 0:
        output[x][index][:] = 1

    # RGB data visualisation
    index = SUBPLOT_RGB * dmap.height + dmap.height - y - 1
    if 0 < vec[0] < dmap.width and 1 < vec[1] < dmap.height and dmap.has_rgb:
        output[x][index][0] = dmap.rgb_array[int(vec[1])][int(vec[0])][0] / 255.0
        output[x][index][1] = dmap.rgb_array[int(vec[1])][int(vec[0])][1] / 255.0
        output[x][index][2] = dmap.rgb_array[int(vec[1])][int(vec[0])][2] / 255.0

    # ensure pixel clipping
    for i in range(SUBPLOT_COUNT):
        index = i * dmap.height + dmap.height - y - 1
        output[x][index][0] = min(max(0, output[x][index][0]), 1)
        output[x][index][1] = min(max(0, output[x][index][1]), 1)
        output[x][index][2] = min(max(0, output[x][index][2]), 1)


def render_plot(dmap: Depthmap):
    # floor and child detection
    floor = dmap.get_floor_level()
    mask, highest = dmap.detect_child(floor)

    # render the visualisations
    output = np.zeros((dmap.width, dmap.height * SUBPLOT_COUNT, 3))
    for x in range(dmap.width):
        for y in range(dmap.height):
            render_pixel(output, x, y, floor, mask, dmap)

    logging.info('height=%fm', highest - floor)
    plt.imshow(output)
