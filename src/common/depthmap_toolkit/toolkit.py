import os
import shutil
import sys
from os import walk
import logging
import logging.config
from pathlib import Path
import functools
from typing import List

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import depthmap
import pcd2depth
import utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


def convert_all_pcds(event, width: int, height: int, calibration: List[List[float]]):
    input_dir = 'export'
    pcd = []
    for _, _, filenames in walk(input_dir):
        pcd = filenames
    pcd.sort()
    try:
        shutil.rmtree('output')
    except BaseException:
        print('no previous data to delete')
    os.makedirs('output/depth')
    # copyfile(input_dir + '/../camera_calibration.txt', 'output/camera_calibration.txt')
    for i in range(len(pcd)):
        depthmap = pcd2depth.process(calibration, input_dir + '/' + pcd[i], width, height)
        pcd2depth.write_depthmap('output/depth/' + pcd[i] + '.depth', depthmap)
    logging.info('Data exported into folder output')


def export_obj(event, width: int, height: int, data: bytes, depth_scale: float, calibration: List[List[float]], max_confidence: float, matrix: list):
    depthmap.export('obj', 'output' + str(index) + '.obj', width, height, data, depth_scale, calibration, max_confidence, matrix)


def export_pcd(event, width: int, height: int, data: bytes, depth_scale: float, calibration: List[List[float]], max_confidence: float, matrix: list):
    depthmap.export('pcd', 'output' + str(index) + '.pcd', width, height, data, depth_scale, calibration, max_confidence, matrix)


def next(event, calibration: List[List[float]], depthmap_dir: str):
    plt.close()
    global index
    index = index + 1
    if (index == size):
        index = 0
    show(depthmap_dir, calibration)


def prev(event, calibration: List[List[float]], depthmap_dir: str):
    plt.close()
    global index
    index = index - 1
    if (index == -1):
        index = size - 1
    show(depthmap_dir, calibration)


def show(depthmap_dir: str, calibration: List[List[float]]):
    if rgb:
        width, height, depth_scale, max_confidence, data, matrix = depthmap.process(plt, depthmap_dir, depth[index], rgb[index])
    else:
        width, height, depth_scale, max_confidence, data, matrix = depthmap.process(plt, depthmap_dir, depth[index], 0)
    angle = depthmap.get_angle_between_camera_and_floor(width, height, calibration)
    logging.info('angle between camera and floor is %f', angle)

    depthmap.show_result(width, height, calibration, data, depth_scale, max_confidence, matrix)
    ax = plt.gca()
    ax.text(0.5, 1.075, depth[index], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    bprev = Button(plt.axes([0.0, 0.0, 0.1, 0.075]), '<<', color='gray')
    bprev.on_clicked(functools.partial(prev, calibration=calibration, depthmap_dir=depthmap_dir))
    bnext = Button(plt.axes([0.9, 0.0, 0.1, 0.075]), '>>', color='gray')
    bnext.on_clicked(functools.partial(next, calibration=calibration, depthmap_dir=depthmap_dir))
    bexport_obj = Button(plt.axes([0.2, 0.0, 0.2, 0.05]), 'Export OBJ', color='gray')
    bexport_obj.on_clicked(functools.partial(export_obj, width=width, height=height, data=data, depth_scale=depth_scale, calibration=calibration, max_confidence=max_confidence, matrix=matrix))
    bexport_pcd = Button(plt.axes([0.4, 0.0, 0.2, 0.05]), 'Export PCD', color='gray')
    bexport_pcd.on_clicked(functools.partial(export_pcd, width=width, height=height, data=data, depth_scale=depth_scale, calibration=calibration, max_confidence=max_confidence, matrix=matrix))
    bconvertPCDs = Button(plt.axes([0.6, 0.0, 0.2, 0.05]), 'Convert all PCDs', color='gray')
    bconvertPCDs.on_clicked(functools.partial(convert_all_pcds, width=width, height=height, calibration=calibration))
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
