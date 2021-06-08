import logging
import logging.config
import math

import functools
import matplotlib.pyplot as plt
import numpy as np
from typing import List

import utils
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


# click on data
last = [0, 0, 0]


def onclick(event, dmap: Depthmap):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.ydata)
        y = dmap.height - int(event.xdata) - 1
        if x > 1 and y > 1 and x < dmap.width - 2 and y < dmap.height - 2:
            depth = dmap.parse_depth(x, y)
            if depth:
                res = dmap.convert_2d_to_3d(1, x, y, depth)
                if res:
                    diff = [last[0] - res[0], last[1] - res[1], last[2] - res[2]]
                    dst = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                    res.append(dst)
                    logging.info('x=%s, y=%s, depth=%s, diff=%s', str(res[0]), str(res[1]), str(res[2]), str(res[3]))
                    last[0] = res[0]
                    last[1] = res[1]
                    last[2] = res[2]
                    return
            logging.info('no valid data')


def detect_child(output: object,
                 x: int,
                 y: int,
                 floor: float,
                 dmap: Depthmap) -> float:

    # highlight the focused child/object using seed algorithm
    highest = floor
    dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    pixel = [int(dmap.width / 2), int(dmap.height / 2)]
    stack = [pixel]
    while len(stack) > 0:

        # get a next pixel from the stack
        pixel = stack.pop()
        depth_center = dmap.parse_depth(pixel[0], pixel[1])

        # add neighbor points (if there is no floor and they are connected)
        index = SUBPLOT_SEGMENTATION * dmap.height + dmap.height - pixel[1] - 1
        if output[pixel[0]][index][2] < 0.1:
            for direction in dirs:
                pixel_dir = [pixel[0] + direction[0], pixel[1] + direction[1]]
                depth_dir = dmap.parse_depth(pixel_dir[0], pixel_dir[1])
                if depth_dir > 0 and (depth_dir - depth_center) < 0.1:
                    stack.append(pixel_dir)

        # update the highest point
        point = dmap.convert_2d_to_3d_oriented(1, pixel[0], pixel[1], depth_center)
        highest = max(highest, point[1])

        # fill the pixels with yellow pattern
        horizontal = ((point[1] - floor) % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
        output[pixel[0]][index][0] = horizontal
        output[pixel[0]][index][1] = horizontal
        output[pixel[0]][index][2] = 0.1

    return highest


def render_pixel(output: object,
                 x: int,
                 y: int,
                 floor: float,
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
    if abs(normal[1]) < 0.5:
        output[x][index][0] = horizontal / (depth * depth)
    if abs(normal[1]) > 0.5:
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
        output[x][index][0] = dmap.im_array[int(vec[1])][int(vec[0])][0] / 255.0
        output[x][index][1] = dmap.im_array[int(vec[1])][int(vec[0])][1] / 255.0
        output[x][index][2] = dmap.im_array[int(vec[1])][int(vec[0])][2] / 255.0

    # ensure pixel clipping
    for i in range(SUBPLOT_COUNT):
        index = i * dmap.height + dmap.height - y - 1
        output[x][index][0] = min(max(0, output[x][index][0]), 1)
        output[x][index][1] = min(max(0, output[x][index][1]), 1)
        output[x][index][2] = min(max(0, output[x][index][2]), 1)


def show(dmap: Depthmap):
    fig = plt.figure()
    fig.canvas.mpl_connect(
        'button_press_event',
        functools.partial(
            onclick,
            dmap=dmap))

    # render the visualisations
    floor = dmap.get_floor_level()
    output = np.zeros((dmap.width, dmap.height * SUBPLOT_COUNT, 3))
    for x in range(dmap.width):
        for y in range(dmap.height):
            render_pixel(output, x, y, floor, dmap)
    highest = detect_child(output, x, y, floor, dmap)

    logging.info('height=%fm', highest - floor)
    plt.imshow(output)
