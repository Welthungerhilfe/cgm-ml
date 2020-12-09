import numpy as np


def quaternion_mult(q: list, r: list) -> list:
    """Multiplication of 2 quaternions"""
    return [r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
            r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
            r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
            r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]]


def point_rotation_by_quaternion(point: list, q: list) -> list:
    """Apply rotation to point in 3D space"""
    r = [0] + point
    q_conj = [q[0], -q[1], -q[2], -q[3]]
    return quaternion_mult(quaternion_mult(q, r), q_conj)[1:]


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
            res = point_rotation_by_quaternion(res, rotation)
            for i in range(0, 2):
                res[i] = res[i] + position[i]
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
        print('Pointcloud exported into ' + filename)


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
        print('Pointcloud exported into ' + filename)


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
        #print(str(calibration[0]) + '\n') #color camera intrinsics - fx, fy, cx, cy
        file.readline()[:-1]
        calibration.append(parse_numbers(file.readline()))
        #print(str(calibration[1]) + '\n') #depth camera intrinsics - fx, fy, cx, cy
        file.readline()[:-1]
        calibration.append(parse_numbers(file.readline()))
        #print(str(calibration[2]) + '\n') #depth camera position relativelly to color camera in meters
        calibration[2][1] *= 8.0  # workaround for wrong calibration data
    return calibration


def parse_confidence(tx, ty):
    """Get confidence of the point in scale 0-1"""
    return data[(int(ty) * width + int(tx)) * 3 + 2] / maxConfidence


def parse_data(filename):
    """Parse depth data"""
    global width, height, depthScale, maxConfidence, data, position, rotation
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
            rotation = (float(header[4]), float(header[5]), float(header[6]), float(header[3]))
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
