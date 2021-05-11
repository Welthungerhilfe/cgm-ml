import os
import shutil
import sys
from os import walk
import logging
import logging.config
from pathlib import Path
import functools

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import depthmap
import pcd2depth
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


def convert_all_pcds(event, width, height, calibration):
    input_dir = 'export'
    pcd = []
    for _, _, filenames in walk(input_dir):
        pcd = filenames
    pcd.sort()
    try:
        shutil.rmtree('output')
    except BaseException:
        print('no previous data to delete')
    os.mkdirs('output/depth')
    # copyfile(input_dir + '/../camera_calibration.txt', 'output/camera_calibration.txt')
    for i in range(len(pcd)):
        depthmap = pcd2depth.process(calibration, input_dir + '/' + pcd[i], width, height)
        pcd2depth.write_depthmap('output/depth/' + pcd[i] + '.depth', depthmap)
    logging.info('Data exported into folder output')


def export_obj(event, height, width, data, depth_scale, calibration, max_confidence):
    depthmap.export('obj', 'output' + str(index) + '.obj', height, width, data, depth_scale, calibration, max_confidence)


def export_pcd(event, height, width, data, depth_scale, calibration, max_confidence):
    depthmap.export('pcd', 'output' + str(index) + '.pcd', height, width, data, depth_scale, calibration, max_confidence)


def next(event, calibration, depthmap_dir):
    plt.close()
    global index
    index = index + 1
    if (index == size):
        index = 0
    show(depthmap_dir, calibration)


def prev(event, calibration, depthmap_dir):
    plt.close()
    global index
    index = index - 1
    if (index == -1):
        index = size - 1
    show(depthmap_dir, calibration)


def show(depthmap_dir, calibration):
    if rgb:
        width, height, depth_scale, max_confidence, data, matrix = depthmap.process(plt, depthmap_dir, depth[index], rgb[index])
    else:
        width, height, depth_scale, max_confidence, data, matrix = depthmap.process(plt, depthmap_dir, depth[index], 0)

    depthmap.show_result(width, height, calibration, data, depth_scale, max_confidence)
    ax = plt.gca()
    ax.text(0.5, 1.075, depth[index], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    bprev = Button(plt.axes([0.0, 0.0, 0.1, 0.075]), '<<', color='gray')
    bprev.on_clicked(functools.partial(prev, calibration=calibration, depthmap_dir=depthmap_dir))
    bnext = Button(plt.axes([0.9, 0.0, 0.1, 0.075]), '>>', color='gray')
    bnext.on_clicked(functools.partial(next, calibration=calibration, depthmap_dir=depthmap_dir))
    bexport_obj = Button(plt.axes([0.2, 0.0, 0.2, 0.05]), 'Export OBJ', color='gray')
    bexport_obj.on_clicked(functools.partial(export_obj, height=height, width=width, data=data, depth_scale=depth_scale, calibration=calibration, max_confidence=max_confidence))
    bexport_pcd = Button(plt.axes([0.4, 0.0, 0.2, 0.05]), 'Export PCD', color='gray')
    bexport_pcd.on_clicked(functools.partial(export_pcd, height=height, width=width, data=data, depth_scale=depth_scale, calibration=calibration, max_confidence=max_confidence))
    bconvertPCDs = Button(plt.axes([0.6, 0.0, 0.2, 0.05]), 'Convert all PCDs', color='gray')
    bconvertPCDs.on_clicked(functools.partial(convert_all_pcds, calibration=calibration))
    plt.show()


if __name__ == "__main__":
    # Prepare
    if len(sys.argv) != 3:
        logging.info('You did not enter depthmap_dir folder and calibration file path')
        logging.info('E.g.: python toolkit.py depthmap_dir calibration_file')
        sys.exit(1)

    depthmap_dir = sys.argv[1]
    calibration_file = sys.argv[2]

    depth = []
    rgb = []
    for (dirpath, dirnames, filenames) in walk(Path(depthmap_dir) / 'depth'):
        depth = filenames
    for (dirpath, dirnames, filenames) in walk(Path(depthmap_dir) / 'rgb'):
        rgb = filenames
    depth.sort()
    rgb.sort()

    calibration = utils.parse_calibration(calibration_file)

    # Make sure there is a new export folder
    try:
        shutil.rmtree('export')
    except BaseException:
        print('no previous data to delete')
    os.mkdir('export')

    # Show viewer
    index = 0
    size = len(depth)
    show(depthmap_dir, calibration)
