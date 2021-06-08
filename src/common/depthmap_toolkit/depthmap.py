from os import stat
import zipfile
import logging
import logging.config
import math

from pathlib import Path
import statistics
from typing import List
import numpy as np
from PIL import Image

import utils
import constants

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


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


    def get_floor_level(self) -> float:
        """Calculate an altitude of the floor in the world coordinates"""
        data = self.data
        width = self.width
        height = self.height
        depth_scale = self.depth_scale
        calibration = self.intrinsics
        max_confidence = self.max_confidence
        matrix = self.matrix

        altitudes = []
        for x in range(width):
            for y in range(height):
                normal = utils.calculate_normal_vector(calibration[1], x, y, width, height, data, depth_scale, matrix)
                if abs(normal[1]) > 0.5:
                    depth = utils.parse_depth(x, y, width, height, data, depth_scale)
                    point = utils.convert_2d_to_3d_oriented(calibration[1], x, y, depth, width, height, matrix)
                    altitudes.append(point[1])
        return statistics.median(altitudes)