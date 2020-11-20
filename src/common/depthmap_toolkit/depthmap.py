import zipfile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import utils

#export data


def export(type, filename):
    if type == 'obj':
        utils.export_obj('export/' + filename, triangulate=True)
    if type == 'pcd':
        utils.export_pcd('export/' + filename)


#click on data
last = [0, 0, 0]


def onclick(event):
    width = utils.getWidth()
    height = utils.getHeight()
    if event.xdata is not None and event.ydata is not None:
        x = int(event.ydata)
        y = height - int(event.xdata) - 1
        if x > 1 and y > 1 and x < width - 2 and y < height - 2:
            depth = utils.parse_depth(x, y)
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


def process(plt, dir, depth, rgb):

    #extract depthmap
    with zipfile.ZipFile(dir + '/depth/' + depth, 'r') as zip_ref:
        zip_ref.extractall('.')
    utils.parse_data('data')

    #read rgb data
    global has_rgb
    global im_array
    if rgb:
        has_rgb = 1
        width = utils.getWidth()
        height = utils.getHeight()
        pil_im = Image.open(dir + '/rgb/' + rgb)
        pil_im = pil_im.resize((width, height), Image.ANTIALIAS)
        im_array = np.asarray(pil_im)
    else:
        has_rgb = 0

    #parse calibration
    global calibration
    calibration = utils.parse_calibration(dir + '/camera_calibration.txt')


def show_result():
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', onclick)
    width = utils.getWidth()
    height = utils.getHeight()
    output = np.zeros((width, height * 3, 3))
    for x in range(width):
        for y in range(height):
            depth = utils.parse_depth(x, y)
            if (depth):
                #convert ToF coordinates into RGB coordinates
                vec = utils.convert2Dto3D(calibration[1], x, y, depth)
                vec[0] += calibration[2][0]
                vec[1] += calibration[2][1]
                vec[2] += calibration[2][2]
                vec = utils.convert_2d_to_3d(calibration[0], vec[0], vec[1], vec[2])

                #set output pixel
                output[x][height - y - 1][:] = 1.0 - min(depth / 2.0, 1.0)  # depth data scaled to be visible
                output[x][height + height - y - 1][:] = utils.parse_confidence(x, y)
                if vec[0] > 0 and vec[1] > 1 and vec[0] < width and vec[1] < height and has_rgb:
                    output[x][2 * height + height - y - 1][0] = im_array[int(vec[1])][int(vec[0])][0] / 255.0
                    output[x][2 * height + height - y - 1][1] = im_array[int(vec[1])][int(vec[0])][1] / 255.0
                    output[x][2 * height + height - y - 1][2] = im_array[int(vec[1])][int(vec[0])][2] / 255.0
    plt.imshow(output)
