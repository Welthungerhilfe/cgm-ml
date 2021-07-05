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


def blur_face(data: np.array, subplot: int, highest: list, dmap: Depthmap) -> np.array:
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
                    index = subplot * dmap.height + dmap.height - ty - 1
                    pixel = pixel + data[tx][index][0]
                    count = count + 1
            index = subplot * dmap.height + dmap.height - y - 1
            output[x][index] = pixel / count

    return output


def render_confidence(output: np.array,
                      subplot: int,
                      dmap: Depthmap):
    for x in range(dmap.width):
        for y in range(dmap.height):
            index = subplot * dmap.height + dmap.height - y - 1
            output[x][index][:] = dmap.parse_confidence(x, y)
            if output[x][index][0] == 0:
                output[x][index][:] = 1


def render_depth(output: np.array,
                 subplot: int,
                 dmap: Depthmap):
    for x in range(dmap.width):
        for y in range(dmap.height):
            depth = dmap.parse_depth(x, y)
            if not depth:
                continue
            index = subplot * dmap.height + dmap.height - y - 1
            output[x][index] = min(max(0, 1.0 - min(depth / 2.0, 1.0)), 1)


def render_normal(output: np.array,
                  subplot: int,
                  dmap: Depthmap):
    for x in range(dmap.width):
        for y in range(dmap.height):
            normal = dmap.calculate_normal_vector(x, y)
            index = subplot * dmap.height + dmap.height - y - 1
            output[x][index][0] = abs(normal[0])
            output[x][index][1] = abs(normal[1])
            output[x][index][2] = abs(normal[2])


def render_rgb(output: np.array,
               subplot: int,
               dmap: Depthmap):
    for x in range(dmap.width):
        for y in range(dmap.height):
            index = subplot * dmap.height + dmap.height - y - 1
            output[x][index][0] = dmap.rgb_array[y][x][0] / 255.0
            output[x][index][1] = dmap.rgb_array[y][x][1] / 255.0
            output[x][index][2] = dmap.rgb_array[y][x][2] / 255.0


def render_segmentation(output: np.array,
                        subplot: int,
                        floor: float,
                        mask: np.array,
                        dmap: Depthmap):
    for x in range(dmap.width):
        for y in range(dmap.height):

            # get depth value
            depth = dmap.parse_depth(x, y)
            if not depth:
                continue

            # segmentation visualisation
            normal = dmap.calculate_normal_vector(x, y)
            point = dmap.convert_2d_to_3d_oriented(1, x, y, depth)
            horizontal = (point[1] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
            vertical_x = (point[0] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
            vertical_z = (point[2] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
            vertical = (vertical_x + vertical_z) / 2.0
            index = subplot * dmap.height + dmap.height - y - 1
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

            # ensure pixel clipping
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
    render_depth(output, SUBPLOT_DEPTH, dmap)
    render_normal(output, SUBPLOT_NORMAL, dmap)
    render_segmentation(output, SUBPLOT_SEGMENTATION, floor, mask, dmap)
    render_confidence(output, SUBPLOT_CONFIDENCE, dmap)
    if dmap.has_rgb:
        render_rgb(output, SUBPLOT_RGB, dmap)
        output = blur_face(output, SUBPLOT_RGB, highest, dmap)

    logging.info('height=%fm', highest[1] - floor)
    return output
