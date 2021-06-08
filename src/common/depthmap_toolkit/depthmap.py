from os import stat
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
from typing import List, Tuple, Union

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


TOOLKIT_DIR = Path(__file__).parents[0].absolute()


def extract_depthmap(depthmap_dir: str, depthmap_fname: str):
    """Extract depthmap from given file"""
    with zipfile.ZipFile(Path(depthmap_dir) / 'depth' / depthmap_fname, 'r') as zip_ref:
        zip_ref.extractall(TOOLKIT_DIR)
    return TOOLKIT_DIR / constants.EXTRACTED_DEPTH_FILE_NAME

class Depthmap:  # Artifact
    """Depthmap
    Args:
        intrinsics ([np.array]): Camera intrinsics
        width ([int]): Width of the depthmap
        height ([int]): Height of the depthmap
        depth_scale: it's in the header of a depthmap file
        data ([bytes]): data TODO rename
        matrix ([type]): not in header
                - position and rotation of the pose
                - pose in different format
    """
    def __init__(self, intrinsics, width, height, data, depth_scale, max_confidence, matrix, rgb_data, has_rgb, im_array):
        self.intrinsics = intrinsics
        self.width = width
        self.height = height
        self.data = data
        self.depth_scale = depth_scale
        self.max_confidence = max_confidence
        self.matrix = matrix
        self.rgb_data = rgb_data
        self.has_rgb = has_rgb
        self.im_array = im_array

    @classmethod
    def create_from_file(cls,
                         depthmap_dir: str,
                         depthmap_fname: str,
                         rgb_fname: str,
                         calibration_file: str):

        # read depthmap data
        path = extract_depthmap(depthmap_dir, depthmap_fname)
        with open(path, 'rb') as f:
            line = f.readline().decode().strip()
            header = line.split('_')
            res = header[0].split('x')
            width = int(res[0])
            height = int(res[1])
            depth_scale = float(header[1])
            max_confidence = float(header[2])
            if len(header) >= 10:
                position = (float(header[7]), float(header[8]), float(header[9]))
                rotation = (float(header[3]), float(header[4]), float(header[5]), float(header[6]))
                matrix = utils.matrix_calculate(position, rotation)
            else:
                matrix = utils.IDENTITY_MATRIX_4D
            data = f.read()
            f.close()

        # read rgb data
        if rgb_fname:
            rgb_data = depthmap_dir + '/rgb/' + rgb_fname
            has_rgb = 1
            pil_im = Image.open(rgb_data)
            pil_im = pil_im.resize((width, height), Image.ANTIALIAS)
            im_array = np.asarray(pil_im)
        else:
            rgb_data = rgb_fname
            has_rgb = 0
            im_array = None

        intrinsics = utils.parse_calibration(calibration_file)

        return cls(intrinsics,
                   width,
                   height,
                   data,
                   depth_scale,
                   max_confidence,
                   matrix,
                   rgb_data,
                   has_rgb,
                   im_array
        )


    def get_angle_between_camera_and_floor(self) -> float:
        """Calculate an angle between camera and floor based on device pose"""
        width = self.width
        height = self.height
        calibration = self.intrinsics
        matrix = self.matrix

        centerx = float(width / 2)
        centery = float(height / 2)
        vector = utils.convert_2d_to_3d_oriented(calibration[1], centerx, centery, 1.0, width, height, matrix)
        angle = 90 + math.degrees(math.atan2(vector[0], vector[1]))
        return angle

    def export(self, type: str, filename: str):
        data = self.data
        width = self.width
        height = self.height
        depth_scale = self.depth_scale
        calibration = self.intrinsics
        max_confidence = self.max_confidence
        matrix = self.matrix

        rgb = self.rgb_data
        if type == 'obj':
            utils.export_obj('export/' + filename, rgb, width, height, data,
                            depth_scale, calibration, matrix, triangulate=True)
        if type == 'pcd':
            utils.export_pcd('export/' + filename, width, height, data, depth_scale, calibration, max_confidence)

    def show_result(self):
        data = self.data
        width = self.width
        height = self.height
        depth_scale = self.depth_scale
        calibration = self.intrinsics
        max_confidence = self.max_confidence
        matrix = self.matrix

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
                render_pixel(output, x, y, floor, self)
        highest = detect_child(output, x, y, floor, self)

        logging.info('height=%fm', highest - floor)
        plt.imshow(output)


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
                 floor: float,
                 dmap: Depthmap) -> float:

    data = dmap.data
    width = dmap.width
    height = dmap.height
    depth_scale = dmap.depth_scale
    calibration = dmap.intrinsics
    max_confidence = dmap.max_confidence
    matrix = dmap.matrix

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
                 floor: float,
                 dmap: Depthmap):
    data = dmap.data
    width = dmap.width
    height = dmap.height
    depth_scale = dmap.depth_scale
    calibration = dmap.intrinsics
    max_confidence = dmap.max_confidence
    matrix = dmap.matrix

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
    if 0 < vec[0] < width and 1 < vec[1] < height and dmap.has_rgb:
        output[x][index][0] = dmap.im_array[int(vec[1])][int(vec[0])][0] / 255.0
        output[x][index][1] = dmap.im_array[int(vec[1])][int(vec[0])][1] / 255.0
        output[x][index][2] = dmap.im_array[int(vec[1])][int(vec[0])][2] / 255.0

    # ensure pixel clipping
    for i in range(SUBPLOT_COUNT):
        index = i * height + height - y - 1
        output[x][index][0] = min(max(0, output[x][index][0]), 1)
        output[x][index][1] = min(max(0, output[x][index][1]), 1)
        output[x][index][2] = min(max(0, output[x][index][2]), 1)
