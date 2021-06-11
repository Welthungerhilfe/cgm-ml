import logging
import logging.config
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


def parse_numbers(line: str) -> List[float]:
    """Parse line of numbers

    Args:
        line: Example: "0.6786797 0.90489584 0.49585155 0.5035042"

    Return:
        numbers: [0.6786797, 0.90489584, 0.49585155, 0.5035042]
    """
    return [float(value) for value in line.split(' ')]
