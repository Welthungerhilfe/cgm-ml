import math
import numpy as np

def add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def cross(a, b):
    output = [0, 0, 0]
    output[0] = a[1] * b[2] - a[2] * b[1]
    output[1] = a[0] * b[2] - a[2] * b[0]
    output[2] = a[0] * b[1] - a[1] * b[0]
    return output


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def normalize(a):
    len = a[0] + a[1] + a[2]
    if len == 0:
        len = 1
    return [a[0] / len, a[1] / len, a[2] / len]


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def length(a, b):
    diff = sub(a, b)
    value = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
    return math.sqrt(value)


def quaternion_mult(q, r):
    return [r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
            r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
            r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
            r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]]


def point_rotation_by_quaternion(point, q):
    r = [0] + point
    q_conj = [q[0], -q[1], -q[2], -q[3]]
    return quaternion_mult(quaternion_mult(q, r), q_conj)[1:]

#convert point into 3D


def convert2Dto3D(intrisics, x, y, z):
    fx = intrisics[0] * float(width)
    fy = intrisics[1] * float(height)
    cx = intrisics[2] * float(width)
    cy = intrisics[3] * float(height)
    tx = (x - cx) * z / fx
    ty = (y - cy) * z / fy
    output = []
    output.append(tx)
    output.append(ty)
    output.append(z)
    return output

#convert point into 3D oriented


def convert2Dto3DOriented(intrisics, x, y, z):
    res = convert2Dto3D(calibration[1], x, y, z)
    if res:
        try:
            res = point_rotation_by_quaternion(res, rotation)
            for i in range(0, 2):
                res[i] = res[i] + position[i]
        except NameError:
            i = 0
    return res

#convert point into 2D


def convert3Dto2D(intrisics, x, y, z):
    fx = intrisics[0] * float(width)
    fy = intrisics[1] * float(height)
    cx = intrisics[2] * float(width)
    cy = intrisics[3] * float(height)
    tx = x * fx / z + cx
    ty = y * fy / z + cy
    output = []
    output.append(tx)
    output.append(ty)
    output.append(z)
    return output

#write obj

#triangulate=True generates OBJ of type mesh
#triangulate=False generates OBJ of type pointcloud
def exportOBJ(filename, triangulate):

    count = 0
    indices = np.zeros((width, height))
    with open(filename, 'w') as file:
        for x in range(2, width - 2):
            for y in range(2, height - 2):
                depth = parseDepth(x, y)
                if depth:
                    res = convert2Dto3DOriented(calibration[1], x, y, depth)
                    if res:
                        count = count + 1
                        indices[x][y] = count  # add index of written vertex into array
                        file.write('v ' + str(res[0]) + ' ' + str(res[1]) + ' ' + str(res[2]) + '\n')

        if triangulate:
            maxDiff = 0.2
            for x in range(2, width - 2):
                for y in range(2, height - 2):
                    #get depth of all points of 2 potential triangles
                    d00 = parseDepth(x, y)
                    d10 = parseDepth(x + 1, y)
                    d01 = parseDepth(x, y + 1)
                    d11 = parseDepth(x + 1, y + 1)

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

#write pcd


def exportPCD(filename):
    with open(filename, 'w') as file:
        count = str(getCount())
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
                depth = parseDepth(x, y)
                if depth:
                    res = convert2Dto3D(calibration[1], x, y, depth)
                    if res:
                        file.write(str(-res[0]) + ' ' + str(res[1]) + ' '
                                   + str(res[2]) + ' ' + str(parseConfidence(x, y)) + '\n')
        print('Pointcloud exported into ' + filename)

#get valid points in depthmaps


def getCount():
    count = 0
    for x in range(2, width - 2):
        for y in range(2, height - 2):
            depth = parseDepth(x, y)
            if depth:
                res = convert2Dto3D(calibration[1], x, y, depth)
                if res:
                    count = count + 1
    return count

#getter


def getWidth():
    return width

#getter


def getHeight():
    return height

#parse calibration file


def parseCalibration(filepath):
    global calibration
    with open(filepath, 'r') as file:
        calibration = []
        file.readline()[:-1]
        calibration.append(parseNumbers(file.readline()))
        #print(str(calibration[0]) + '\n') #color camera intrinsics - fx, fy, cx, cy
        file.readline()[:-1]
        calibration.append(parseNumbers(file.readline()))
        #print(str(calibration[1]) + '\n') #depth camera intrinsics - fx, fy, cx, cy
        file.readline()[:-1]
        calibration.append(parseNumbers(file.readline()))
        #print(str(calibration[2]) + '\n') #depth camera position relativelly to color camera in meters
        calibration[2][1] *= 8.0  # workaround for wrong calibration data
    return calibration

#get confidence of the point in scale 0-1


def parseConfidence(tx, ty):
    return ord(data[(int(ty) * width + int(tx)) * 3 + 2]) / maxConfidence

#parse depth data


def parseData(filename):
    global width, height, depthScale, maxConfidence, data, position, rotation
    with open('data', 'rb') as file:
        line = str(file.readline())[:-1]
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

#get depth of the point in meters


def parseDepth(tx, ty):
    depth = ord(data[(int(ty) * width + int(tx)) * 3 + 0]) << 8
    depth += ord(data[(int(ty) * width + int(tx)) * 3 + 1])
    depth *= depthScale
    return depth

#get smoothed depth of the point in meters


def parseDepthSmoothed(tx, ty, s):
    center = parseDepth(tx, ty)
    count = 1
    depth = center
    for x in range(tx - s, tx + s):
        for y in range(ty - s, ty + s):
            value = parseDepth(x, y)
            if abs(center - value) < 0.1:
                depth = depth + value
                count = count + 1
    return depth / count

#parse line of numbers


def parseNumbers(line):
    output = []
    values = line.split(' ')
    for value in values:
        output.append(float(value))
    return output

#parse PCD


def parsePCD(filepath):
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
                values = parseNumbers(line)
                data.append(values)
    return data

#parse line of values


def parseValues(line):
    output = []
    values = line.split(' ')
    for value in values:
        output.append(value)
    return output

#setter


def setWidth(value):
    global width
    width = value

#setter


def setHeight(value):
    global height
    height = value
