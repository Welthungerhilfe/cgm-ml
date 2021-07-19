import zipfile
import math
import sys

from pathlib import Path
import statistics
import numpy as np
from PIL import Image
from typing import List

from depthmap_utils import (
    matrix_calculate, IDENTITY_MATRIX_4D, parse_numbers, diff, cross, norm, matrix_transform_point)
from constants import EXTRACTED_DEPTH_FILE_NAME, MASK_FLOOR, MASK_CHILD, MASK_INVALID


TOOLKIT_DIR = Path(__file__).parents[0].absolute()


def extract_depthmap(depthmap_dir: str, depthmap_fname: str):
    """Extract depthmap from given file"""
    with zipfile.ZipFile(Path(depthmap_dir) / 'depth' / depthmap_fname, 'r') as zip_ref:
        zip_ref.extractall(TOOLKIT_DIR)
    return TOOLKIT_DIR / EXTRACTED_DEPTH_FILE_NAME


class Depthmap:
    """Depthmap and optional RGB

    Args:
        intrinsic (np.array): Camera intrinsic
        width (int): Width of the depthmap
        height (int): Height of the depthmap
        data (bytes): pixel_data
        depth_scale (float): Scalar to scale depthmap pixel to meters
        max_confidence (float): Confidence is amount of IR light reflected
                                (e.g. 0 to 255 in Lenovo, new standard is 0 to 7)
                                This is actually an int.
        matrix (list): Header contains a pose (= position and rotation)
                       - matrix is a list representation of this pose
                       - can be used to project into a different space
        rgb_fpath (str): Path to RGB file (e.g. to the jpg)
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
            rgb_fpath,
            rgb_array):
        self.intrinsics = intrinsics
        self.width = width
        self.height = height
        self.data = data
        self.depth_scale = depth_scale
        self.max_confidence = max_confidence
        self.matrix = matrix
        self.rgb_fpath = rgb_fpath
        self.rgb_array = rgb_array

    @property
    def has_rgb(self) -> bool:
        """Bool that indicates if the object has RGB data"""
        return self.rgb_array is not None

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
            rgb_fpath = depthmap_dir + '/rgb/' + rgb_fname
            pil_im = Image.open(rgb_fpath)
            pil_im = pil_im.resize((width, height), Image.ANTIALIAS)
            rgb_array = np.asarray(pil_im)
        else:
            rgb_fpath = rgb_fname
            rgb_array = None

        # read calibration file
        intrinsics = parse_calibration(calibration_file)

        return cls(intrinsics,
                   width,
                   height,
                   data,
                   depth_scale,
                   max_confidence,
                   matrix,
                   rgb_fpath,
                   rgb_array
                   )

    def calculate_normal_vector(self, x: float, y: float) -> list:
        """Calculate normal vector of depthmap point based on neighbors"""

        # Get depth of the neighbor pixels
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

    def detect_child(self, floor: float) -> np.array:

        mask, segments = self.detect_objects(floor)

        # Select the most focused segment
        closest = sys.maxsize
        focus = -1
        for segment in segments:
            a = segment[1][0] - int(self.width / 2)
            b = segment[1][1] - int(self.height / 2)
            c = segment[1][2] - int(self.width / 2)
            d = segment[1][3] - int(self.height / 2)
            distance = a * a + b * b + c * c + d * d
            if closest > distance:
                closest = distance
                focus = segment[0]

        mask = np.where(mask == focus, MASK_CHILD, mask)

        return mask

    def detect_floor(self, floor: float) -> np.array:
        mask = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                depth = self.parse_depth_smoothed(x, y)
                if not depth:
                    mask[x][y] = MASK_INVALID
                    continue
                normal = self.calculate_normal_vector(x, y)
                point = self.convert_2d_to_3d_oriented(1, x, y, depth)
                if abs(normal[1]) > 0.5 and abs(point[1] - floor) < 0.1:
                    mask[x][y] = MASK_FLOOR

        return mask

    def detect_objects(self, floor: float) -> [np.array, list]:

        # Detect objects/children using seed algorithm
        current = -1
        segments = []
        dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        mask = self.detect_floor(floor)
        for x in range(self.width):
            for y in range(self.height):
                if mask[x][y] != 0:
                    continue
                pixel = [x, y]
                aabb = [pixel[0], pixel[1], pixel[0], pixel[1]]
                stack = [pixel]
                while len(stack) > 0:

                    # Get a next pixel from the stack
                    pixel = stack.pop()
                    depth_center = self.parse_depth(pixel[0], pixel[1])

                    # Add neighbor points (if there is no floor and they are connected)
                    if mask[pixel[0]][pixel[1]] == 0:
                        for direction in dirs:
                            pixel_dir = [pixel[0] + direction[0], pixel[1] + direction[1]]
                            depth_dir = self.parse_depth(pixel_dir[0], pixel_dir[1])
                            if depth_dir > 0 and abs(depth_dir - depth_center) < 0.1:
                                stack.append(pixel_dir)

                    # Update AABB
                    aabb[0] = min(pixel[0], aabb[0])
                    aabb[1] = min(pixel[1], aabb[1])
                    aabb[2] = max(pixel[0], aabb[2])
                    aabb[3] = max(pixel[1], aabb[3])

                    # Update the mask
                    mask[pixel[0]][pixel[1]] = current

                # Check if the object size is valid
                object_size_pixels = max(aabb[2] - aabb[0], aabb[3] - aabb[1])
                if object_size_pixels > self.width / 4:
                    segments.append([current, aabb])
                current = current - 1

        return mask, segments

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

    def get_highest_point(self, mask: np.array) -> list:
        highest = [-sys.maxsize, -sys.maxsize, -sys.maxsize]
        for x in range(self.width):
            for y in range(self.height):
                if mask[x][y] == MASK_CHILD:
                    depth = self.parse_depth(x, y)
                    point = self.convert_2d_to_3d_oriented(1, x, y, depth)
                    if highest[1] < point[1]:
                        highest = point
        return highest

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

        # Get all neighbor depths
        depth_center = self.parse_depth(tx, ty)
        depth_x_minus = self.parse_depth(tx - 1, ty)
        depth_x_plus = self.parse_depth(tx + 1, ty)
        depth_y_minus = self.parse_depth(tx, ty - 1)
        depth_y_plus = self.parse_depth(tx, ty + 1)

        # Ensure the depth is defined
        if 0 in [depth_center, depth_x_plus, depth_y_minus, depth_y_plus]:
            return 0

        # Average the depth value
        depths = [depth_x_minus, depth_x_plus, depth_y_minus, depth_y_plus, depth_center]
        return sum(depths) / len(depths)

    def convert_3d_to_2d(self, sensor: int, x: float, y: float, depth: float) -> list:
        """Convert point in meters into point in pixels

        Args:
            sensor: Tells if this is ToF sensor or RGB sensor
            x: X-pos in m
            y: Y-pos in m
            depth: distance from sensor to object at (x, y)

        Returns:
            tx, ty, depth
        """
        return convert_3d_to_2d(self.intrinsics[sensor], x, y, depth, self.width, self.height)


def convert_3d_to_2d(intrinsics: list, x: float, y: float, depth: float, width: int, height: int):
    fx = intrinsics[0] * float(width)
    fy = intrinsics[1] * float(height)
    cx = intrinsics[2] * float(width)
    cy = intrinsics[3] * float(height)
    tx = x * fx / depth + cx
    ty = y * fy / depth + cy
    return [tx, ty, depth]


def parse_calibration(filepath: str) -> List[List[float]]:
    """Parse calibration file
    filepath: The content of a calibration file looks like this:
        Color camera intrinsic:
        0.6786797 0.90489584 0.49585155 0.5035042
        Depth camera intrinsic:
        0.6786797 0.90489584 0.49585155 0.5035042
    """
    with open(filepath, 'r') as f:
        calibration = []
        for _ in range(2):
            f.readline().strip()
            line_with_numbers = f.readline()
            intrinsic = parse_numbers(line_with_numbers)
            calibration.append(intrinsic)
    return calibration
