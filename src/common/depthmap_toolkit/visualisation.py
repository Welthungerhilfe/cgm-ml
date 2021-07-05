import logging
import logging.config

import numpy as np

from constants import MASK_CHILD
from depthmap import Depthmap
from depthmap_utils import diff, length

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

CHILD_HEAD_HEIGHT_IN_METERS = 0.25
PATTERN_LENGTH_IN_METERS = 0.1

SUBPLOT_DEPTH = 0
SUBPLOT_NORMAL = 1
SUBPLOT_SEGMENTATION = 2
SUBPLOT_CONFIDENCE = 3
SUBPLOT_RGB = 4
SUBPLOT_COUNT = 5


def blur_face(data: np.array, highest: list, dmap: Depthmap) -> np.array:
    """Faceblur of the detected standing child.

    It uses the highest point of the child and blur all pixels in distance less than CHILD_HEAD_HEIGHT_IN_METERS.
    """

    # copy values
    output = np.copy(data)

    # blur RGB data around face
    for x in range(dmap.width):
        for y in range(dmap.height):

            # count distance from the highest child point
            depth = dmap.parse_depth(x, y)
            if not depth:
                continue
            point = dmap.convert_2d_to_3d_oriented(1, x, y, depth)
            distance = length(diff(point, highest))
            if distance >= CHILD_HEAD_HEIGHT_IN_METERS:
                continue

            # Gausian blur
            pixel = [0, 0, 0]
            count = 0
            step = 5
            for tx in range(x - step, x + step):
                for ty in range(y - step, y + step):
                    if not (0 < tx < dmap.width and 0 < ty < dmap.height):
                        continue
                    index = SUBPLOT_RGB * dmap.height + dmap.height - ty - 1
                    pixel = pixel + data[tx][index][0]
                    count = count + 1
            index = SUBPLOT_RGB * dmap.height + dmap.height - y - 1
            output[x][index] = pixel / count

    return output


def render_pixel(output: np.array,
                 x: int,
                 y: int,
                 floor: float,
                 mask: np.array,
                 dmap: Depthmap):

    # RGB data visualisation
    if dmap.has_rgb:
        index = SUBPLOT_RGB * dmap.height + dmap.height - y - 1
        output[x][index][0] = dmap.rgb_array[y][x][0] / 255.0
        output[x][index][1] = dmap.rgb_array[y][x][1] / 255.0
        output[x][index][2] = dmap.rgb_array[y][x][2] / 255.0

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

    # ensure pixel clipping
    for i in range(SUBPLOT_COUNT):
        index = i * dmap.height + dmap.height - y - 1
        output[x][index][0] = min(max(0, output[x][index][0]), 1)
        output[x][index][1] = min(max(0, output[x][index][1]), 1)
        output[x][index][2] = min(max(0, output[x][index][2]), 1)


def render_plot(dmap: Depthmap) -> np.array:
    # floor and child detection
    floor = dmap.get_floor_level()
    mask = dmap.detect_child(floor)
    highest = dmap.get_highest_point(mask)

    # render the visualisations
    output = np.zeros((dmap.width, dmap.height * SUBPLOT_COUNT, 3))
    for x in range(dmap.width):
        for y in range(dmap.height):
            render_pixel(output, x, y, floor, mask, dmap)
    if dmap.has_rgb:
        output = blur_face(output, highest, dmap)

    logging.info('height=%fm', highest[1] - floor)
    return output
