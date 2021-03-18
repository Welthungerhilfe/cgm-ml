import logging
import logging.config
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


def matrix_calculate(position: list, rotation: list) -> list:
    """Calculate a matrix image->world from device position and rotation"""

    output = [1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1]

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

    output[12] = position[0]
    output[13] = position[1]
    output[14] = position[2]
    return output


def matrix_transform_point(point: list, matrix: list) -> list:
    """Transformation of point by matrix"""
    output = [ 0, 0, 0, 1 ]
    output[0] = point[0] * matrix[0] + point[1] * matrix[4] + point[2] * matrix[8] + matrix[12]
    output[1] = point[0] * matrix[1] + point[1] * matrix[5] + point[2] * matrix[9] + matrix[13]
    output[2] = point[0] * matrix[2] + point[1] * matrix[6] + point[2] * matrix[10] + matrix[14]
    output[3] = point[0] * matrix[3] + point[1] * matrix[7] + point[2] * matrix[11] + matrix[15]

    output[0] /= abs(output[3])
    output[1] /= abs(output[3])
    output[2] /= abs(output[3])
    output[3] = 1
    return output

def convert2Dto3D(intrisics: list, x: float, y: float, z: float) -> list:
    """Convert point in pixels into point in meters"""
    fx = intrisics[0] * float(width)
    fy = intrisics[1] * float(height)
    cx = intrisics[2] * float(width)
    cy = intrisics[3] * float(height)
    tx = (x - cx) * z / fx
    ty = (y - cy) * z / fy
    return [tx, ty, z]


def convert_2d_to_3d_oriented(intrisics: list, x: float, y: float, z: float) -> list:
    """Convert point in pixels into point in meters (applying rotation)"""
    res = convert2Dto3D(calibration[1], x, y, z)
    if res:
        try:
            res = [-res[0], -res[1], res[2]]
            res = matrix_transform_point(res, matrix)
        except NameError:
            i = 0
    return res


def convert_2d_to_3d(intrisics: list, x: float, y: float, z: float) -> list:
    """Convert point in meters into point in pixels"""
    fx = intrisics[0] * float(width)
    fy = intrisics[1] * float(height)
    cx = intrisics[2] * float(width)
    cy = intrisics[3] * float(height)
    tx = x * fx / z + cx
    ty = y * fy / z + cy
    return [tx, ty, z]


def export_obj(filename, triangulate):
    """

    triangulate=True generates OBJ of type mesh
    triangulate=False generates OBJ of type pointcloud
    """
    count = 0
    indices = np.zeros((width, height))
    with open(filename, 'w') as file:
        for x in range(2, width - 2):
            for y in range(2, height - 2):
                depth = parse_depth(x, y)
                if depth:
                    res = convert_2d_to_3d_oriented(calibration[1], x, y, depth)
                    if res:
                        count = count + 1
                        indices[x][y] = count  # add index of written vertex into array
                        file.write('v ' + str(res[0]) + ' ' + str(res[1]) + ' ' + str(res[2]) + '\n')

        if triangulate:
            maxDiff = 0.2
            for x in range(2, width - 2):
                for y in range(2, height - 2):
                    #get depth of all points of 2 potential triangles
                    d00 = parse_depth(x, y)
                    d10 = parse_depth(x + 1, y)
                    d01 = parse_depth(x, y + 1)
                    d11 = parse_depth(x + 1, y + 1)

                    #check if first triangle points have existing indices
                    if indices[x][y] > 0 and indices[x + 1][y] > 0 and indices[x][y + 1] > 0:
                        #check if the triangle size is valid (to prevent generating triangle connecting child and background)
                        if abs(d00 - d10) + abs(d00 - d01) + abs(d10 - d01) < maxDiff:
                            file.write('f ' + str(int(indices[x][y])) + ' ' + str(int(indices[x + 1][y])) + ' ' + str(int(indices[x][y + 1])) + '\n')

                    #check if second triangle points have existing indices
                    if indices[x + 1][y + 1] > 0 and indices[x + 1][y] > 0 and indices[x][y + 1] > 0:
                        #check if the triangle size is valid (to prevent generating triangle connecting child and background)
                        if abs(d11 - d10) + abs(d11 - d01) + abs(d10 - d01) < maxDiff:
                            file.write('f ' + str(int(indices[x + 1][y + 1])) + ' ' + str(int(indices[x + 1][y])) + ' ' + str(int(indices[x][y + 1])) + '\n')
        logging.info('Mesh exported into %s', filename)


def export_pcd(filename):
    with open(filename, 'w') as file:
        count = str(_get_count())
        file.write('# timestamp 1 1 float 0\n')
        file.write('# .PCD v.7 - Point Cloud Data file format\n')
        file.write('VERSION .7\n')
        file.write('FIELDS x y z c\n')
        file.write('SIZE 4 4 4 4\n')
        file.write('TYPE F F F F\n')
        file.write('COUNT 1 1 1 1\n')
        file.write('WIDTH ' + count + '\n')
        file.write('HEIGHT 1\n')
        file.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        file.write('POINTS ' + count + '\n')
        file.write('DATA ascii\n')
        for x in range(2, width - 2):
            for y in range(2, height - 2):
                depth = parse_depth(x, y)
                if depth:
                    res = convert2Dto3D(calibration[1], x, y, depth)
                    if res:
                        file.write(str(-res[0]) + ' ' + str(res[1]) + ' '
                                   + str(res[2]) + ' ' + str(parse_confidence(x, y)) + '\n')
        logging.info('Pointcloud exported into %s', filename)


def _get_count():
    count = 0
    for x in range(2, width - 2):
        for y in range(2, height - 2):
            depth = parse_depth(x, y)
            if depth:
                res = convert2Dto3D(calibration[1], x, y, depth)
                if res:
                    count = count + 1
    return count


def parse_calibration(filepath):
    """Parse calibration file"""
    global calibration
    with open(filepath, 'r') as file:
        calibration = []
        file.readline()[:-1]
        calibration.append(parse_numbers(file.readline()))
        #logging.info(str(calibration[0]) + '\n') #color camera intrinsics - fx, fy, cx, cy
        file.readline()[:-1]
        calibration.append(parse_numbers(file.readline()))
        #logging.info(str(calibration[1]) + '\n') #depth camera intrinsics - fx, fy, cx, cy
        file.readline()[:-1]
        calibration.append(parse_numbers(file.readline()))
        #logging.info(str(calibration[2]) + '\n') #depth camera position relativelly to color camera in meters
        calibration[2][1] *= 8.0  # workaround for wrong calibration data
    return calibration


def parse_confidence(tx, ty):
    """Get confidence of the point in scale 0-1"""
    return data[(int(ty) * width + int(tx)) * 3 + 2] / maxConfidence


def parse_data(filename):
    """Parse depth data"""
    global width, height, depthScale, maxConfidence, data, matrix
    with open('data', 'rb') as file:
        line = file.readline().decode().strip()
        header = line.split('_')
        res = header[0].split('x')
        width = int(res[0])
        height = int(res[1])
        depthScale = float(header[1])
        maxConfidence = float(header[2])
        if len(header) >= 10:
            position = (float(header[7]), float(header[8]), float(header[9]))
            rotation = (float(header[3]), float(header[4]), float(header[5]), float(header[6]))
            matrix = matrix_calculate(position, rotation)
        data = file.read()
        file.close()


def parse_depth(tx, ty):
    """Get depth of the point in meters"""
    depth = data[(int(ty) * width + int(tx)) * 3 + 0] << 8
    depth += data[(int(ty) * width + int(tx)) * 3 + 1]
    depth *= depthScale
    return depth


def parse_numbers(line):
    """Parse line of numbers"""
    output = []
    values = line.split(' ')
    for value in values:
        output.append(float(value))
    return output


def parse_pcd(filepath):
    with open(filepath, 'r') as file:
        data = []
        while True:
            line = str(file.readline())
            if line.startswith('DATA'):
                break

        while True:
            line = str(file.readline())
            if not line:
                break
            else:
                values = parse_numbers(line)
                data.append(values)
    return data


def getWidth():
    return width


def getHeight():
    return height


def setWidth(value):
    global width
    width = value


def setHeight(value):
    global height
    height = value
