import zipfile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import utils

#export data


def export(type, filename):
    if type == 'obj':
        utils.exportOBJ('export/' + filename, triangulate=True)
    if type == 'pcd':
        utils.exportPCD('export/' + filename)


#click on data
last = [0, 0, 0]


def onclick(event):
    width = utils.getWidth()
    height = utils.getHeight()
    if event.xdata is not None and event.ydata is not None:
        x = width - int(event.ydata) - 1
        y = height - int(event.xdata) - 1
        if x > 1 and y > 1 and x < width - 2 and y < height - 2:
            depth = utils.parseDepth(x, y)
            if depth:
                res = utils.convert2Dto3D(calibration[1], x, y, depth)
                if res:
                    diff = [last[0] - res[0], last[1] - res[1], last[2] - res[2]]
                    dst = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                    res.append(dst)
                    print('x=' + str(res[0]) + ', y=' + str(res[1])
                          + ', depth=' + str(res[2]) + ', diff=' + str(res[3]))
                    last[0] = res[0]
                    last[1] = res[1]
                    last[2] = res[2]
                    return
            print('no valid data')

#get input


def process(plt, dir, depth, rgb):

    #extract depthmap
    with zipfile.ZipFile(dir + '/depth/' + depth, 'r') as zip_ref:
        zip_ref.extractall('.')
    utils.parseData('data')

    #read rgb data
    global hasRGB
    global im_array
    if rgb:
        hasRGB = 1
        width = utils.getWidth()
        height = utils.getHeight()
        pil_im = Image.open(dir + '/rgb/' + rgb)
        pil_im = pil_im.resize((width, height), Image.ANTIALIAS)
        im_array = np.asarray(pil_im)
    else:
        hasRGB = 0

    #parse calibration
    global calibration
    calibration = utils.parseCalibration(dir + '/camera_calibration.txt')

#show edges


def showEdges():
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', onclick)

    #calculate smoothed normalmap
    smoothing = 3
    maxdepth = 0
    width = utils.getWidth()
    height = utils.getHeight()
    normalmap = np.zeros((width, height, 3))
    for x in range(1 + smoothing, width - 1 - smoothing):
        for y in range(1 + smoothing, height - 1 - smoothing):
            maxdepth = max(maxdepth, utils.parseDepth(x, y))
            cmx = utils.convert2Dto3D(calibration[1], x - 1, y, utils.parseDepthSmoothed(x - 1, y, smoothing))
            cmy = utils.convert2Dto3D(calibration[1], x, y - 1, utils.parseDepthSmoothed(x, y - 1, smoothing))
            cpx = utils.convert2Dto3D(calibration[1], x + 1, y, utils.parseDepthSmoothed(x + 1, y, smoothing))
            cpy = utils.convert2Dto3D(calibration[1], x, y + 1, utils.parseDepthSmoothed(x, y + 1, smoothing))
            n = utils.normalize(utils.cross(utils.sub(cpx, cmx), utils.sub(cpy, cmy)))
            normalmap[x][height - y - 1][0] = min(abs(n[0]), 1)
            normalmap[x][height - y - 1][1] = min(abs(n[1]), 1)
            normalmap[x][height - y - 1][2] = min(abs(n[2]), 1)

    #fade normalmap
    #for x in range(1, width):
    #  for y in range(1, height):
    #    factor = utils.parseDepth(x, y) / maxdepth
    #    normalmap[x][height - y - 1][0] *= (1 - factor)
    #    normalmap[x][height - y - 1][1] *= (1 - factor)
    #    normalmap[x][height - y - 1][2] *= (1 - factor)

    #calculate edgemap
    edgemap = np.zeros((width, height, 3))
    for x in range(2, width - 2):
        for y in range(2, height - 2):

            #calculate edge from depthmap
            d = utils.convert2Dto3D(calibration[1], x, y, utils.parseDepth(x, y))
            mx = utils.convert2Dto3D(calibration[1], x - 1, y, utils.parseDepth(x - 1, y))
            my = utils.convert2Dto3D(calibration[1], x, y - 1, utils.parseDepth(x, y - 1))
            px = utils.convert2Dto3D(calibration[1], x + 1, y, utils.parseDepth(x + 1, y))
            py = utils.convert2Dto3D(calibration[1], x, y + 1, utils.parseDepth(x, y + 1))
            edge = max(max(utils.length(d, mx), utils.length(d, my)), max(utils.length(d, px), utils.length(d, py)))
            if edge > 0.1:  # difference of depth in meters
                edgemap[x][height - y - 1][1] = 1

            #calculate edge from normalmap
            d = normalmap[x][height - y - 1]
            mx = normalmap[x - 1][height - y - 1]
            my = normalmap[x][height - y - 1 - 1]
            px = normalmap[x + 1][height - y - 1]
            py = normalmap[x][height - y - 1 + 1]
            edge = max(max(utils.dot(d, mx), utils.dot(d, my)), max(utils.dot(d, px), utils.dot(d, py)))
            if edge > 0.9:  # normal matching in percentage
                edgemap[x][height - y - 1][0] = 1

    #calculate output
    output = np.zeros((width, height, 3))
    for x in range(2, width - 2):
        for y in range(2, height - 2):
            if edgemap[x][height - y - 1][1] == 1:
                output[x][height - y - 1][2] = 1
            else:
                if edgemap[x][height - y - 1][0] == 0:
                    if edgemap[x - 1][height - y - 1][0] == 1:
                        output[x][height - y - 1][1] = 1
                    if edgemap[x][height - y - 1 - 1][0] == 1:
                        output[x][height - y - 1][1] = 1
                    if edgemap[x + 1][height - y - 1][0] == 1:
                        output[x][height - y - 1][1] = 1
                    if edgemap[x][height - y - 1 + 1][0] == 1:
                        output[x][height - y - 1][1] = 1

    #plt.imshow(normalmap, extent=[0, height, 0, width])
    #plt.imshow(edgemap, extent=[0, height, 0, width])
    plt.imshow(output, extent=[0, height, 0, width])

#show result


def showResult():
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', onclick)
    width = utils.getWidth()
    height = utils.getHeight()
    output = np.zeros((width, height * 3, 3))
    for x in range(width):
        for y in range(height):
            depth = utils.parseDepth(x, y)
            if (depth):
                #convert ToF coordinates into RGB coordinates
                vec = utils.convert2Dto3D(calibration[1], x, y, depth)
                vec[0] += calibration[2][0]
                vec[1] += calibration[2][1]
                vec[2] += calibration[2][2]
                vec = utils.convert3Dto2D(calibration[0], vec[0], vec[1], vec[2])

                #set output pixel
                output[x][height - y - 1][:] = 1.0 - min(depth / 2.0, 1.0)  # depth data scaled to be visible
                output[x][height + height - y - 1][:] = utils.parseConfidence(x, y)
                if vec[0] > 0 and vec[1] > 1 and vec[0] < width and vec[1] < height and hasRGB:
                    output[x][2 * height + height - y - 1][0] = im_array[int(vec[1])][int(vec[0])][0] / 255.0
                    output[x][2 * height + height - y - 1][1] = im_array[int(vec[1])][int(vec[0])][1] / 255.0
                    output[x][2 * height + height - y - 1][2] = im_array[int(vec[1])][int(vec[0])][2] / 255.0
    plt.imshow(output)
