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

PATTERN_LENGTH_IN_METERS = 0.1

SUBPLOT_DEPTH = 0
SUBPLOT_NORMAL = 1
SUBPLOT_SEGMENTATION = 2
SUBPLOT_CONFIDENCE = 3
SUBPLOT_RGB = 4
SUBPLOT_COUNT = 5


def export(type: str, filename: str, width: int, height: int, data: bytes, depth_scale: float,
           calibration: List[List[float]], max_confidence: float, matrix: list):
    rgb = CURRENT_RGB
    if type == 'obj':
        utils.export_obj('export/' + filename, rgb, width, height, data,
                         depth_scale, calibration, matrix, triangulate=True)
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


def get_angle_between_camera_and_floor(width: int, height: int, calibration: List[List[float]], matrix: list) -> float:
    """Calculate an angle between camera and floor based on device pose"""
    centerx = float(width / 2)
    centery = float(height / 2)
    vector = utils.convert_2d_to_3d_oriented(calibration[1], centerx, centery, 1.0, width, height, matrix)
    angle = 90 + math.degrees(math.atan2(vector[0], vector[1]))
    return angle


def get_floor_level(width: int,
                    height: int,
                    calibration: List[List[float]],
                    data: bytes,
                    depth_scale: float,
                    max_confidence: float,
                    matrix: list) -> float:
    """Calculate an altitude of the floor in the world coordinates"""
    altitudes = []
    for x in range(width):
        for y in range(height):
            normal = utils.calculate_normal_vector(calibration[1], x, y, width, height, data, depth_scale, matrix)
            if abs(normal[1]) > 0.5:
                depth = utils.parse_depth(x, y, width, height, data, depth_scale)
                point = utils.convert_2d_to_3d_oriented(calibration[1], x, y, depth, width, height, matrix)
                altitudes.append(point[1])
    return statistics.median(altitudes)


def detect_child(output: object,
                 x: int,
                 y: int,
                 width: int,
                 height: int,
                 calibration: List[List[float]],
                 data: bytes,
                 depth_scale: float,
                 max_confidence: float,
                 matrix: list,
                 floor: float) -> float:

    # highlight the focused child/object using seed algorithm
    highest = floor
    dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    pixel = [int(width / 2), int(height / 2)]
    stack = [pixel]
    while len(stack) > 0:

        # get a next pixel from the stack
        pixel = stack.pop()
        depth_center = utils.parse_depth(pixel[0], pixel[1], width, height, data, depth_scale)

        # add neighbor points (if there is no floor and they are connected)
        index = SUBPLOT_SEGMENTATION * height + height - pixel[1] - 1
        if output[pixel[0]][index][2] < 0.1:
            for direction in dirs:
                pixel_dir = [pixel[0] + direction[0], pixel[1] + direction[1]]
                depth_dir = utils.parse_depth(pixel_dir[0], pixel_dir[1], width, height, data, depth_scale)
                if depth_dir > 0 and (depth_dir - depth_center) < 0.1:
                    stack.append(pixel_dir)

        # update the highest point
        point = utils.convert_2d_to_3d_oriented(calibration[1], pixel[0], pixel[1], depth_center, width, height, matrix)
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
                 width: int,
                 height: int,
                 calibration: List[List[float]],
                 data: bytes,
                 depth_scale: float,
                 max_confidence: float,
                 matrix: list,
                 floor: float):
    depth = utils.parse_depth(x, y, width, height, data, depth_scale)
    if not depth:
        return

    # convert ToF coordinates into RGB coordinates
    vec = utils.convert_2d_to_3d(calibration[1], x, y, depth, width, height)
    vec[0] += calibration[2][0]
    vec[1] += calibration[2][1]
    vec[2] += calibration[2][2]
    vec = utils.convert_3d_to_2d(calibration[0], vec[0], vec[1], vec[2], width, height)

    # depth data scaled to be visible
    index = SUBPLOT_DEPTH * height + height - y - 1
    output[x][index] = 1.0 - min(depth / 2.0, 1.0)

    # get 3d point and normal vector
    point = utils.convert_2d_to_3d_oriented(calibration[1], x, y, depth, width, height, matrix)
    normal = utils.calculate_normal_vector(calibration[1], x, y, width, height, data, depth_scale, matrix)
    index = SUBPLOT_NORMAL * height + height - y - 1
    output[x][index][0] = abs(normal[0])
    output[x][index][1] = abs(normal[1])
    output[x][index][2] = abs(normal[2])

    # world coordinates visualisation
    horizontal = (point[1] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
    vertical_x = (point[0] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
    vertical_z = (point[2] % PATTERN_LENGTH_IN_METERS) / PATTERN_LENGTH_IN_METERS
    vertical = (vertical_x + vertical_z) / 2.0
    index = SUBPLOT_SEGMENTATION * height + height - y - 1
    if abs(normal[1]) < 0.5:
        output[x][index][0] = horizontal / (depth * depth)
    if abs(normal[1]) > 0.5:
        if abs(point[1] - floor) < 0.1:
            output[x][index][2] = vertical / (depth * depth)
        else:
            output[x][index][1] = vertical / (depth * depth)

    # confidence value
    index = SUBPLOT_CONFIDENCE * height + height - y - 1
    output[x][index][:] = utils.parse_confidence(x, y, data, max_confidence, width)
    if output[x][index][0] == 0:
        output[x][index][:] = 1

    # RGB data
    index = SUBPLOT_RGB * height + height - y - 1
    if 0 < vec[0] < width and 1 < vec[1] < height and HAS_RGB:
        output[x][index][0] = IM_ARRAY[int(vec[1])][int(vec[0])][0] / 255.0
        output[x][index][1] = IM_ARRAY[int(vec[1])][int(vec[0])][1] / 255.0
        output[x][index][2] = IM_ARRAY[int(vec[1])][int(vec[0])][2] / 255.0

    # ensure pixel clipping
    for i in range(SUBPLOT_COUNT):
        index = i * height + height - y - 1
        output[x][index][0] = min(max(0, output[x][index][0]), 1)
        output[x][index][1] = min(max(0, output[x][index][1]), 1)
        output[x][index][2] = min(max(0, output[x][index][2]), 1)


def show_result(width: int,
                height: int,
                calibration: List[List[float]],
                data: bytes,
                depth_scale: float,
                max_confidence: float,
                matrix: list):
    fig = plt.figure()
    fig.canvas.mpl_connect(
        'button_press_event',
        functools.partial(
            onclick,
            width=width,
            height=height,
            data=data,
            depth_scale=depth_scale,
            calibration=calibration))

    # render the visualisations
    floor = get_floor_level(width, height, calibration, data, depth_scale, max_confidence, matrix)
    output = np.zeros((width, height * SUBPLOT_COUNT, 3))
    for x in range(width):
        for y in range(height):
            render_pixel(output, x, y, width, height, calibration, data, depth_scale, max_confidence, matrix, floor)
    highest = detect_child(output, x, y, width, height, calibration, data, depth_scale, max_confidence, matrix, floor)

    logging.info('height=%fm', highest - floor)
    plt.imshow(output)
