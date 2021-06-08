import logging
import logging.config
import numpy as np
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

IDENTITY_MATRIX_4D = [1., 0., 0., 0.,
                      0., 1., 0., 0.,
                      0., 0., 1., 0.,
                      0., 0., 0., 1.]


def cross(a: list, b: list) -> list:
    """Cross product of two vectors"""
    c = [a[1] * b[2] - a[2] * b[1],
         a[2] * b[0] - a[0] * b[2],
         a[0] * b[1] - a[1] * b[0]]
    return c


def diff(a: list, b: list) -> list:
    """Difference of two vectors"""
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def norm(v: list) -> list:
    """Vector normalize"""
    length = abs(v[0]) + abs(v[1]) + abs(v[2])
    if length == 0:
        length = 1
    return [v[0] / length, v[1] / length, v[2] / length]


def matrix_calculate(position: list, rotation: list) -> list:
    """Calculate a matrix image->world from device position and rotation"""

    output = IDENTITY_MATRIX_4D

    sqw = rotation[3] * rotation[3]
    sqx = rotation[0] * rotation[0]
    sqy = rotation[1] * rotation[1]
    sqz = rotation[2] * rotation[2]

    invs = 1 / (sqx + sqy + sqz + sqw)
    output[0] = (sqx - sqy - sqz + sqw) * invs
    output[5] = (-sqx + sqy - sqz + sqw) * invs
    output[10] = (-sqx - sqy + sqz + sqw) * invs

    tmp1 = rotation[0] * rotation[1]
    tmp2 = rotation[2] * rotation[3]
    output[1] = 2.0 * (tmp1 + tmp2) * invs
    output[4] = 2.0 * (tmp1 - tmp2) * invs

    tmp1 = rotation[0] * rotation[2]
    tmp2 = rotation[1] * rotation[3]
    output[2] = 2.0 * (tmp1 - tmp2) * invs
    output[8] = 2.0 * (tmp1 + tmp2) * invs

    tmp1 = rotation[1] * rotation[2]
    tmp2 = rotation[0] * rotation[3]
    output[6] = 2.0 * (tmp1 + tmp2) * invs
    output[9] = 2.0 * (tmp1 - tmp2) * invs

    output[12] = -position[0]
    output[13] = -position[1]
    output[14] = -position[2]
    return output


def matrix_transform_point(point: list, matrix: list) -> list:
    """Transformation of point by matrix"""
    output = [0, 0, 0, 1]
    output[0] = point[0] * matrix[0] + point[1] * matrix[4] + point[2] * matrix[8] + matrix[12]
    output[1] = point[0] * matrix[1] + point[1] * matrix[5] + point[2] * matrix[9] + matrix[13]
    output[2] = point[0] * matrix[2] + point[1] * matrix[6] + point[2] * matrix[10] + matrix[14]
    output[3] = point[0] * matrix[3] + point[1] * matrix[7] + point[2] * matrix[11] + matrix[15]

    output[0] /= abs(output[3])
    output[1] /= abs(output[3])
    output[2] /= abs(output[3])
    output[3] = 1
    return output


def convert_2d_to_3d(intrisics: list, x: float, y: float, z: float, width: int, height: int) -> list:
    """Convert point in pixels into point in meters"""
    fx = intrisics[0] * float(width)
    fy = intrisics[1] * float(height)
    cx = intrisics[2] * float(width)
    cy = intrisics[3] * float(height)
    tx = (x - cx) * z / fx
    ty = (y - cy) * z / fy
    return [tx, ty, z]


def convert_2d_to_3d_oriented(intrisics: list, x: float, y: float, z: float,
                              width: int, height: int, matrix: list) -> list:
    """Convert point in pixels into point in meters (applying rotation)"""
    res = convert_2d_to_3d(intrisics, x, y, z, width, height)
    if not res:
        return res

    # special case for Google Tango devices with different rotation
    if width == 180 and height == 135:
        res = [res[0], -res[1], res[2]]
    else:
        res = [-res[0], res[1], res[2]]
    try:
        res = matrix_transform_point(res, matrix)
        res = [res[0], -res[1], res[2]]
    except NameError:
        pass
    return res


def convert_3d_to_2d(intrisics: list, x: float, y: float, z: float, width: int, height: int) -> list:
    """Convert point in meters into point in pixels"""
    fx = intrisics[0] * float(width)
    fy = intrisics[1] * float(height)
    cx = intrisics[2] * float(width)
    cy = intrisics[3] * float(height)
    tx = x * fx / z + cx
    ty = y * fy / z + cy
    return [tx, ty, z]


def export_obj(filename: str,
               rgb: bool,
               width: int,
               height: int,
               data: bytes,
               depth_scale: float,
               calibration: List[List[float]],
               matrix: list,
               triangulate: bool,
               ):
    """Export .obj file, which can be visualized in tools like Meshlab.

    triangulate=True generates OBJ of type mesh
    triangulate=False generates OBJ of type pointcloud
    rgb=path to color frame
    """
    count = 0
    indices = np.zeros((width, height))

    # Create MTL file (a standart extension of OBJ files to define geometry materials and textures)
    material = filename[:len(filename) - 4] + '.mtl'
    if rgb:
        with open(material, 'w') as f:
            f.write('newmtl default\n')
            f.write('map_Kd ../' + rgb + '\n')

    with open(filename, 'w') as f:
        if rgb:
            f.write('mtllib ' + material[filename.index('/') + 1:] + '\n')
            f.write('usemtl default\n')
        for x in range(2, width - 2):
            for y in range(2, height - 2):
                depth = parse_depth(x, y, width, height, data, depth_scale)
                if not depth:
                    continue
                res = convert_2d_to_3d_oriented(calibration[1], x, y, depth, width, height, matrix)
                if not res:
                    continue
                count = count + 1
                indices[x][y] = count  # add index of written vertex into array
                f.write('v ' + str(res[0]) + ' ' + str(res[1]) + ' ' + str(res[2]) + '\n')
                f.write('vt ' + str(x / width) + ' ' + str(1 - y / height) + '\n')

        if triangulate:
            do_triangulation(width, height, data, depth_scale, indices, f)
        logging.info('Mesh exported into %s', filename)


def do_triangulation(width, height, data, depth_scale, indices, filehandle):
    max_diff = 0.2
    for x in range(2, width - 2):
        for y in range(2, height - 2):
            # get depth of all points of 2 potential triangles
            d00 = parse_depth(x, y, width, height, data, depth_scale)
            d10 = parse_depth(x + 1, y, width, height, data, depth_scale)
            d01 = parse_depth(x, y + 1, width, height, data, depth_scale)
            d11 = parse_depth(x + 1, y + 1, width, height, data, depth_scale)

            # check if first triangle points have existing indices
            if indices[x][y] > 0 and indices[x + 1][y] > 0 and indices[x][y + 1] > 0:
                # check if the triangle size is valid (to prevent generating triangle
                # connecting child and background)
                if abs(d00 - d10) + abs(d00 - d01) + abs(d10 - d01) < max_diff:
                    c = str(int(indices[x][y]))
                    b = str(int(indices[x + 1][y]))
                    a = str(int(indices[x][y + 1]))
                    # define triangle indices in (world coordinates / texture coordinates)
                    filehandle.write('f ' + a + '/' + a + ' ' + b + '/' + b + ' ' + c + '/' + c + '\n')

            # check if second triangle points have existing indices
            if indices[x + 1][y + 1] > 0 and indices[x + 1][y] > 0 and indices[x][y + 1] > 0:
                # check if the triangle size is valid (to prevent generating triangle
                # connecting child and background)
                if abs(d11 - d10) + abs(d11 - d01) + abs(d10 - d01) < max_diff:
                    a = str(int(indices[x + 1][y + 1]))
                    b = str(int(indices[x + 1][y]))
                    c = str(int(indices[x][y + 1]))
                    # define triangle indices in (world coordinates / texture coordinates)
                    filehandle.write('f ' + a + '/' + a + ' ' + b + '/' + b + ' ' + c + '/' + c + '\n')


def write_pcd_header(filehandle, count):
    filehandle.write('# timestamp 1 1 float 0\n')
    filehandle.write('# .PCD v.7 - Point Cloud Data file format\n')
    filehandle.write('VERSION .7\n')
    filehandle.write('FIELDS x y z c\n')
    filehandle.write('SIZE 4 4 4 4\n')
    filehandle.write('TYPE F F F F\n')
    filehandle.write('COUNT 1 1 1 1\n')
    filehandle.write('WIDTH ' + count + '\n')
    filehandle.write('HEIGHT 1\n')
    filehandle.write('VIEWPOINT 0 0 0 1 0 0 0\n')
    filehandle.write('POINTS ' + count + '\n')
    filehandle.write('DATA ascii\n')


def export_pcd(filename: str, width: int, height: int, data: bytes, depth_scale: float,
               calibration: List[List[float]], max_confidence: float):
    with open(filename, 'w') as f:
        count = str(_get_count(width, height, data, depth_scale, calibration))
        write_pcd_header(f, count)

        for x in range(2, width - 2):
            for y in range(2, height - 2):
                depth = parse_depth(x, y, width, height, data, depth_scale)
                if not depth:
                    continue
                res = convert_2d_to_3d(calibration[1], x, y, depth, width, height)
                if not res:
                    continue
                f.write(str(-res[0]) + ' ' + str(res[1]) + ' '
                        + str(res[2]) + ' ' + str(parse_confidence(x, y, data, max_confidence, width)) + '\n')
        logging.info('Pointcloud exported into %s', filename)


def _get_count(width: int, height: int, data: bytes, depth_scale: float, calibration: List[List[float]]) -> int:
    count = 0
    for x in range(2, width - 2):
        for y in range(2, height - 2):
            depth = parse_depth(x, y, width, height, data, depth_scale)
            if not depth:
                continue
            res = convert_2d_to_3d(calibration[1], x, y, depth, width, height)
            if not res:
                continue
            count = count + 1
    return count


def parse_calibration(filepath: str) -> List[List[float]]:
    """Parse calibration file"""
    with open(filepath, 'r') as f:
        calibration = []
        for _ in range(3):
            f.readline()[:-1]
            calibration.append(parse_numbers(f.readline()))
        calibration[2][1] *= 8.0  # workaround for wrong calibration data
    return calibration


def parse_confidence(tx: int, ty: int, data: bytes, max_confidence: float, width: int):
    """Get confidence of the point in scale 0-1"""
    return data[(int(ty) * width + int(tx)) * 3 + 2] / max_confidence


def parse_data(filename: str):
    """Parse depth data"""
    with open(filename, 'rb') as f:
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

    return data, width, height, depth_scale, max_confidence, matrix


def parse_depth(tx: int, ty: int, width: int, height: int, data: bytes, depth_scale: float) -> float:
    """Get depth of the point in meters"""
    if tx < 1 or ty < 1 or tx >= width or ty >= height:
        return 0.
    depth = data[(int(ty) * width + int(tx)) * 3 + 0] << 8
    depth += data[(int(ty) * width + int(tx)) * 3 + 1]
    depth *= depth_scale
    return depth


def parse_depth_smoothed(tx: int, ty: int, width: int, height: int, data: bytes, depth_scale: float) -> float:
    """Get average depth value from neighboring pixels"""
    depth_center = parse_depth(tx, ty, width, height, data, depth_scale)
    depth_x_minus = parse_depth(tx - 1, ty, width, height, data, depth_scale)
    depth_x_plus = parse_depth(tx + 1, ty, width, height, data, depth_scale)
    depth_y_minus = parse_depth(tx, ty - 1, width, height, data, depth_scale)
    depth_y_plus = parse_depth(tx, ty + 1, width, height, data, depth_scale)
    depths = [depth_x_minus, depth_x_plus, depth_y_minus, depth_y_plus, depth_center]
    return sum(depths) / len(depths)


def parse_numbers(line: str) -> List[float]:
    """Parse line of numbers"""
    return [float(value) for value in line.split(' ')]


def parse_pcd(filepath: str) -> List[List[float]]:
    with open(filepath, 'r') as f:
        data = []
        while True:
            line = str(f.readline())
            if line.startswith('DATA'):
                break

        while True:
            line = str(f.readline())
            if not line:
                break
            else:
                values = parse_numbers(line)
                data.append(values)
    return data
