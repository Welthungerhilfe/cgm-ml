import zipfile
import logging
import logging.config
import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import functools
import statistics
from typing import List

import utils
import constants

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

SUBPLOT_DEPTH = 0
SUBPLOT_NORMAL = 1
SUBPLOT_SEGMENTATION = 2
SUBPLOT_CONFIDENCE = 3
SUBPLOT_RGB = 4
SUBPLOT_COUNT = 5


def export(type: str, filename: str, width: int, height: int, data: bytes, depth_scale: float, calibration: List[List[float]], max_confidence: float, matrix: list):
    rgb = CURRENT_RGB
    if type == 'obj':
        utils.export_obj('export/' + filename, rgb, width, height, data, depth_scale, calibration, matrix, triangulate=True)
    if type == 'pcd':
        utils.export_pcd('export/' + filename, width, height, data, depth_scale, calibration, max_confidence)


# click on data
last = [0, 0, 0]


def onclick(event, width: int, height: int, data: bytes, depth_scale: float, calibration: List[List[float]]):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.ydata)
        y = height - int(event.xdata) - 1
        if x > 1 and y > 1 and x < width - 2 and y < height - 2:
            depth = utils.parse_depth(x, y, width, height, data, depth_scale)
            if depth:
                res = utils.convert_2d_to_3d(calibration[1], x, y, depth, width, height)
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


def extract_depthmap(dir_path: str, filename: str):
    """Extract depthmap from given file"""
    with zipfile.ZipFile(Path(dir_path) / 'depth' / filename, 'r') as zip_ref:
        zip_ref.extractall('.')


def process(dir_path: str, depth: str, rgb: str):

    extract_depthmap(dir_path, depth)

    data, width, height, depth_scale, max_confidence, matrix = utils.parse_data(constants.EXTRACTED_DEPTH_FILE_NAME)

    # read rgb data
    global CURRENT_RGB
    global HAS_RGB
    global IM_ARRAY
    if rgb:
        CURRENT_RGB = dir_path + '/rgb/' + rgb
        HAS_RGB = 1
        pil_im = Image.open(CURRENT_RGB)
        pil_im = pil_im.resize((width, height), Image.ANTIALIAS)
        IM_ARRAY = np.asarray(pil_im)
    else:
        CURRENT_RGB = rgb
        HAS_RGB = 0

    return width, height, depth_scale, max_confidence, data, matrix


def get_angle_between_camera_and_floor(width: int, height: int, calibration: List[List[float]], matrix: list):
    centerx = float(width / 2)
    centery = float(height / 2)
    vector = utils.convert_2d_to_3d_oriented(calibration[1], centerx, centery, 1.0, width, height, matrix)
    angle = 90 + math.degrees(math.atan2(vector[0], vector[1]))
    return angle


def get_floor_level(width: int, height: int, calibration: List[List[float]], data: bytes, depth_scale: float, max_confidence: float, matrix: list):
    altitudes = []
    for x in range(width):
        for y in range(height):
            depth = utils.parse_depth(x, y, width, height, data, depth_scale)
            v = utils.convert_2d_to_3d_oriented(calibration[1], x, y, depth, width, height, matrix)
            xm = utils.convert_2d_to_3d_oriented(calibration[1], x - 1, y, utils.parse_depth_smoothed(x - 1, y, width, height, data, depth_scale), width, height, matrix)
            xp = utils.convert_2d_to_3d_oriented(calibration[1], x + 1, y, utils.parse_depth_smoothed(x + 1, y, width, height, data, depth_scale), width, height, matrix)
            yp = utils.convert_2d_to_3d_oriented(calibration[1], x, y + 1, utils.parse_depth_smoothed(x, y + 1, width, height, data, depth_scale), width, height, matrix)
            n = utils.norm(utils.cross(utils.diff(yp, xm), utils.diff(yp, xp)))
            if abs(n[1]) > 0.5:
                altitudes.append(v[1])
    return statistics.median(altitudes)


def render_pixel(output: object, x: int, y: int, width: int, height: int, calibration: List[List[float]], data: bytes, depth_scale: float, max_confidence: float, matrix: list, floor: float):
    depth = utils.parse_depth(x, y, width, height, data, depth_scale)
    if (depth):
        # convert ToF coordinates into RGB coordinates
        vec = utils.convert_2d_to_3d(calibration[1], x, y, depth, width, height)
        vec[0] += calibration[2][0]
        vec[1] += calibration[2][1]
        vec[2] += calibration[2][2]
        vec = utils.convert_3d_to_2d(calibration[0], vec[0], vec[1], vec[2], width, height)

        # depth data scaled to be visible
        output[SUBPLOT_DEPTH * height + x][height - y - 1] = 1.0 - min(depth / 2.0, 1.0)

        # depth data normal
        v = utils.convert_2d_to_3d_oriented(calibration[1], x, y, depth, width, height, matrix)
        xm = utils.convert_2d_to_3d_oriented(calibration[1], x - 1, y, utils.parse_depth_smoothed(x - 1, y, width, height, data, depth_scale), width, height, matrix)
        xp = utils.convert_2d_to_3d_oriented(calibration[1], x + 1, y, utils.parse_depth_smoothed(x + 1, y, width, height, data, depth_scale), width, height, matrix)
        yp = utils.convert_2d_to_3d_oriented(calibration[1], x, y + 1, utils.parse_depth_smoothed(x, y + 1, width, height, data, depth_scale), width, height, matrix)
        n = utils.norm(utils.cross(utils.diff(yp, xm), utils.diff(yp, xp)))
        output[x][SUBPLOT_NORMAL * height + height - y - 1][0] = abs(n[0])
        output[x][SUBPLOT_NORMAL * height + height - y - 1][1] = abs(n[1])
        output[x][SUBPLOT_NORMAL * height + height - y - 1][2] = abs(n[2])

        # world coordinates visualisation
        horizontal = (v[1] % 0.1) * 10
        vertical = (v[0] % 0.1) * 5 + (v[2] % 0.1) * 5
        if abs(n[1]) < 0.5:
            output[x][SUBPLOT_SEGMENTATION * height + height - y - 1][0] = horizontal / (depth * depth)
        if abs(n[1]) > 0.5:
            if abs(v[1] - floor) < 0.1:
                output[x][SUBPLOT_SEGMENTATION * height + height - y - 1][2] = vertical / (depth * depth)
            else:
                output[x][SUBPLOT_SEGMENTATION * height + height - y - 1][1] = vertical / (depth * depth)

        # confidence value
        output[x][SUBPLOT_CONFIDENCE * height + height - y - 1][:] = utils.parse_confidence(x, y, data, max_confidence, width)
        if output[x][SUBPLOT_CONFIDENCE * height + height - y - 1][0] == 0:
            output[x][SUBPLOT_CONFIDENCE * height + height - y - 1][:] = 1

        # RGB data
        if vec[0] > 0 and vec[1] > 1 and vec[0] < width and vec[1] < height and HAS_RGB:
            output[x][SUBPLOT_RGB * height + height - y - 1][0] = IM_ARRAY[int(vec[1])][int(vec[0])][0] / 255.0
            output[x][SUBPLOT_RGB * height + height - y - 1][1] = IM_ARRAY[int(vec[1])][int(vec[0])][1] / 255.0
            output[x][SUBPLOT_RGB * height + height - y - 1][2] = IM_ARRAY[int(vec[1])][int(vec[0])][2] / 255.0

        # ensure pixel clipping
        for i in range(SUBPLOT_COUNT):
            output[x][i * height + height - y - 1][0] = min(max(0, output[x][i * height + height - y - 1][0]), 1)
            output[x][i * height + height - y - 1][1] = min(max(0, output[x][i * height + height - y - 1][1]), 1)
            output[x][i * height + height - y - 1][2] = min(max(0, output[x][i * height + height - y - 1][2]), 1)


def show_result(width: int, height: int, calibration: List[List[float]], data: bytes, depth_scale: float, max_confidence: float, matrix: list):
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', functools.partial(onclick, width=width, height=height, data=data, depth_scale=depth_scale, calibration=calibration))

    floor = get_floor_level(width, height, calibration, data, depth_scale, max_confidence, matrix)
    output = np.zeros((width, height * SUBPLOT_COUNT, 3))
    for x in range(width):
        for y in range(height):
            render_pixel(output, x, y, width, height, calibration, data, depth_scale, max_confidence, matrix, floor)

    #highlight the focused child/object using seed algorithm
    highest = floor
    dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    stack = []
    p = [int(width / 2), int(height / 2)]
    stack.append(p)
    while (len(stack) > 0):
        p = stack.pop()
        depth = utils.parse_depth(p[0], p[1], width, height, data, depth_scale)
        if output[p[0]][SUBPLOT_SEGMENTATION * height + height - p[1] - 1][2] < 0.1:
            for dir in dirs:
                t = [p[0] + dir[0], p[1] + dir[1]]
                d = utils.parse_depth(t[0], t[1], width, height, data, depth_scale)
                if d > 0 and (d - depth) < 0.1:
                    stack.append(t)
        v = utils.convert_2d_to_3d_oriented(calibration[1], p[0], p[1], depth, width, height, matrix)
        if highest < v[1]:
            highest = v[1]
        horizontal = ((v[1] - floor) % 0.1) * 10
        output[p[0]][SUBPLOT_SEGMENTATION * height + height - p[1] - 1][0] = horizontal
        output[p[0]][SUBPLOT_SEGMENTATION * height + height - p[1] - 1][1] = horizontal
        output[p[0]][SUBPLOT_SEGMENTATION * height + height - p[1] - 1][2] = 0.1

    logging.info('height=%fm', highest - floor)
    plt.imshow(output)
