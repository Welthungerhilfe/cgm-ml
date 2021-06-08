import zipfile
import logging
import logging.config
import math

from pathlib import Path
import statistics
import numpy as np
from PIL import Image

from utils import (
    matrix_calculate, IDENTITY_MATRIX_4D, parse_numbers, diff, cross, norm, matrix_transform_point)
from constants import EXTRACTED_DEPTH_FILE_NAME

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


TOOLKIT_DIR = Path(__file__).parents[0].absolute()


def extract_depthmap(depthmap_dir: str, depthmap_fname: str):
    """Extract depthmap from given file"""
    with zipfile.ZipFile(Path(depthmap_dir) / 'depth' / depthmap_fname, 'r') as zip_ref:
        zip_ref.extractall(TOOLKIT_DIR)
    return TOOLKIT_DIR / EXTRACTED_DEPTH_FILE_NAME


class Depthmap:
    """Depthmap and RGB TODO consider renaming to Artifact

    Args:
        intrinsics (np.array): Camera intrinsics
        width (int): Width of the depthmap
        height (int): Height of the depthmap
        data (bytes): pixel_data
        depth_scale (float): ??? (it's in the header of a depthmap file)
        max_confidence (float): ???
        matrix (list): not in header
                - position and rotation of the pose
                - pose in different format
        rgb_data (str): Path to RGB file (e.g. to the jpg)
        has_rgb (bool): Flag to indicate if the artifact has RGB data
        rgb_array (np.array): RGB data
    """
    def __init__(
            self,
            intrinsics,
            width,
            height,
            data,
            depth_scale,
            max_confidence,
            matrix,
            rgb_data,
            has_rgb,
            rgb_array):
        self.intrinsics = intrinsics
        self.width = width
        self.height = height
        self.data = data
        self.depth_scale = depth_scale
        self.max_confidence = max_confidence
        self.matrix = matrix
        self.rgb_data = rgb_data
        self.has_rgb = has_rgb
        self.rgb_array = rgb_array

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
                matrix = matrix_calculate(position, rotation)
            else:
                matrix = IDENTITY_MATRIX_4D
            data = f.read()
            f.close()

        # read rgb data
        if rgb_fname:
            rgb_data = depthmap_dir + '/rgb/' + rgb_fname
            has_rgb = True
            pil_im = Image.open(rgb_data)
            pil_im = pil_im.resize((width, height), Image.ANTIALIAS)
            rgb_array = np.asarray(pil_im)
        else:
            rgb_data = rgb_fname
            has_rgb = False
            rgb_array = None

        # read calibration file
        with open(calibration_file, 'r') as f:
            intrinsics = []
            for _ in range(2):
                f.readline().strip()
                line_with_numbers = f.readline()
                intrinsic = parse_numbers(line_with_numbers)
                intrinsics.append(intrinsic)

        return cls(intrinsics,
                   width,
                   height,
                   data,
                   depth_scale,
                   max_confidence,
                   matrix,
                   rgb_data,
                   has_rgb,
                   rgb_array
                   )

    def calculate_normal_vector(self, x: float, y: float) -> list:
        """Calculate normal vector of depthmap point based on neightbors"""

        # Get depth of the neightbor pixels
        depth_center = self.parse_depth_smoothed(x, y)
        depth_x_minus = self.parse_depth_smoothed(x - 1, y)
        depth_y_minus = self.parse_depth_smoothed(x, y - 1)

        # Create a triangle from neighbor points
        point_a = self.convert_2d_to_3d_oriented(1, x, y, depth_center)
        point_b = self.convert_2d_to_3d_oriented(1, x - 1, y, depth_x_minus)
        point_c = self.convert_2d_to_3d_oriented(1, x, y - 1, depth_y_minus)

        # Calculate a normal of the triangle
        vector_u = diff(point_a, point_b)
        vector_v = diff(point_a, point_c)
        normal = cross(vector_u, vector_v)

        # Ensure the normal has a length of one
        return norm(normal)

    def convert_2d_to_3d(self, sensor: int, x: float, y: float, depth: float) -> list:
        """Convert point in pixels into point in meters"""
        fx = self.intrinsics[sensor][0] * float(self.width)
        fy = self.intrinsics[sensor][1] * float(self.height)
        cx = self.intrinsics[sensor][2] * float(self.width)
        cy = self.intrinsics[sensor][3] * float(self.height)
        tx = (x - cx) * depth / fx
        ty = (y - cy) * depth / fy
        return [tx, ty, depth]

    def convert_2d_to_3d_oriented(self, sensor: int, x: float, y: float, depth: float) -> list:
        """Convert point in pixels into point in meters (applying rotation)"""
        res = self.convert_2d_to_3d(sensor, x, y, depth)
        if not res:
            return res

        # special case for Google Tango devices with different rotation
        if self.width == 180 and self.height == 135:
            res = [res[0], -res[1], res[2]]
        else:
            res = [-res[0], res[1], res[2]]
        try:
            res = matrix_transform_point(res, self.matrix)
            res = [res[0], -res[1], res[2]]
        except NameError:
            pass
        return res

    def convert_3d_to_2d(self, sensor: int, x: float, y: float, depth: float) -> list:
        """Convert point in meters into point in pixels"""
        fx = self.intrinsics[sensor][0] * float(self.width)
        fy = self.intrinsics[sensor][1] * float(self.height)
        cx = self.intrinsics[sensor][2] * float(self.width)
        cy = self.intrinsics[sensor][3] * float(self.height)
        tx = x * fx / depth + cx
        ty = y * fy / depth + cy
        return [tx, ty, depth]

    def get_angle_between_camera_and_floor(self) -> float:
        """Calculate an angle between camera and floor based on device pose"""
        centerx = float(self.width / 2)
        centery = float(self.height / 2)
        vector = self.convert_2d_to_3d_oriented(1, centerx, centery, 1.0)
        angle = 90 + math.degrees(math.atan2(vector[0], vector[1]))
        return angle

    def get_floor_level(self) -> float:
        """Calculate an altitude of the floor in the world coordinates"""
        altitudes = []
        for x in range(self.width):
            for y in range(self.height):
                normal = self.calculate_normal_vector(x, y)
                if abs(normal[1]) > 0.5:
                    depth = self.parse_depth(x, y)
                    point = self.convert_2d_to_3d_oriented(1, x, y, depth)
                    altitudes.append(point[1])
        return statistics.median(altitudes)

    def parse_confidence(self, tx: int, ty):
        """Get confidence of the point in scale 0-1"""
        return self.data[(int(ty) * self.width + int(tx)) * 3 + 2] / self.max_confidence

    def parse_depth(self, tx: int, ty: int) -> float:
        """Get depth of the point in meters"""
        if tx < 1 or ty < 1 or tx >= self.width or ty >= self.height:
            return 0.
        depth = self.data[(int(ty) * self.width + int(tx)) * 3 + 0] << 8
        depth += self.data[(int(ty) * self.width + int(tx)) * 3 + 1]
        depth *= self.depth_scale
        return depth

    def parse_depth_smoothed(self, tx: int, ty) -> float:
        """Get average depth value from neighboring pixels"""
        depth_center = self.parse_depth(tx, ty)
        depth_x_minus = self.parse_depth(tx - 1, ty)
        depth_x_plus = self.parse_depth(tx + 1, ty)
        depth_y_minus = self.parse_depth(tx, ty - 1)
        depth_y_plus = self.parse_depth(tx, ty + 1)
        depths = [depth_x_minus, depth_x_plus, depth_y_minus, depth_y_plus, depth_center]
        return sum(depths) / len(depths)
