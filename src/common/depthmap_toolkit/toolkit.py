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
import utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


def export_obj(event,
               width: int,
               height: int,
               data: bytes,
               depth_scale: float,
               calibration: List[List[float]],
               max_confidence: float,
               matrix: list,
               ):
    fname = f'output{index}.obj'
    depthmap.export('obj', fname, width, height, data, depth_scale, calibration, max_confidence, matrix)


def export_pcd(event, width: int, height: int, data: bytes, depth_scale: float,
               calibration: List[List[float]], max_confidence: float, matrix: list):
    fname = f'output{index}.pcd'
    depthmap.export('pcd', fname, width, height, data, depth_scale, calibration, max_confidence, matrix)


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
    rgb_filename = rgb_filenames[index] if rgb_filenames else 0
    width, height, depth_scale, max_confidence, data, matrix = depthmap.process(
        depthmap_dir, depth_filenames[index], rgb_filename)
    angle = depthmap.get_angle_between_camera_and_floor(width, height, calibration, matrix)
    logging.info('angle between camera and floor is %f', angle)

    depthmap.show_result(width, height, calibration, data, depth_scale, max_confidence, matrix)
    ax = plt.gca()
    ax.text(
        0.5,
        1.075,
        depth_filenames[index],
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes)
    bprev = Button(plt.axes([0.0, 0.0, 0.1, 0.075]), '<<', color='gray')
    bprev.on_clicked(functools.partial(prev, calibration=calibration, depthmap_dir=depthmap_dir))
    bnext = Button(plt.axes([0.9, 0.0, 0.1, 0.075]), '>>', color='gray')
    bnext.on_clicked(functools.partial(next, calibration=calibration, depthmap_dir=depthmap_dir))
    bexport_obj = Button(plt.axes([0.3, 0.0, 0.2, 0.05]), 'Export OBJ', color='gray')
    bexport_obj.on_clicked(
        functools.partial(
            export_obj,
            width=width,
            height=height,
            data=data,
            depth_scale=depth_scale,
            calibration=calibration,
            max_confidence=max_confidence,
            matrix=matrix))
    bexport_pcd = Button(plt.axes([0.5, 0.0, 0.2, 0.05]), 'Export PCD', color='gray')
    bexport_pcd.on_clicked(
        functools.partial(
            export_pcd,
            width=width,
            height=height,
            data=data,
            depth_scale=depth_scale,
            calibration=calibration,
            max_confidence=max_confidence,
            matrix=matrix))
    plt.show()


if __name__ == "__main__":
    # Prepare
    if len(sys.argv) != 3:
        logging.info('You did not enter depthmap_dir folder and calibration file path')
        logging.info('E.g.: python toolkit.py depthmap_dir calibration_file')
        sys.exit(1)

    depthmap_dir = sys.argv[1]
    calibration_file = sys.argv[2]

    depth_filenames = []
    for (dirpath, dirnames, filenames) in walk(Path(depthmap_dir) / 'depth'):
        depth_filenames.extend(filenames)
    depth_filenames.sort()

    rgb_filenames = []
    for (dirpath, dirnames, filenames) in walk(Path(depthmap_dir) / 'rgb'):
        rgb_filenames.extend(filenames)
    rgb_filenames.sort()

    calibration = utils.parse_calibration(calibration_file)

    # Clear export folder
    try:
        shutil.rmtree('export')
    except BaseException:
        print('no previous data to delete')
    os.mkdir('export')

    # Show viewer
    index = 0
    size = len(depth_filenames)
    show(depthmap_dir, calibration)
