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


def export_obj(event, dmap):
    fname = f'output{index}.obj'
    dmap.export('obj', fname)


def export_pcd(event, dmap):
    fname = f'output{index}.pcd'
    dmap.export('pcd', fname)


def next(event, calibration_file: str, depthmap_dir: str):
    plt.close()
    global index
    index = index + 1
    if (index == size):
        index = 0
    show(depthmap_dir, calibration_file)


def prev(event, calibration_file: str, depthmap_dir: str):
    plt.close()
    global index
    index = index - 1
    if (index == -1):
        index = size - 1
    show(depthmap_dir, calibration_file)


def show(depthmap_dir: str, calibration_file: str):
    rgb_filename = rgb_filenames[index] if rgb_filenames else 0
    dmap = depthmap.Depthmap.create_from_file(depthmap_dir, depth_filenames[index], rgb_filename)

    angle = dmap.get_angle_between_camera_and_floor()
    logging.info('angle between camera and floor is %f', angle)

    dmap.show_result()
    ax = plt.gca()
    ax.text(
        0.5,
        1.075,
        depth_filenames[index],
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes)
    bprev = Button(plt.axes([0.0, 0.0, 0.1, 0.075]), '<<', color='gray')
    bprev.on_clicked(functools.partial(prev, calibration_file=calibration_file, depthmap_dir=depthmap_dir))
    bnext = Button(plt.axes([0.9, 0.0, 0.1, 0.075]), '>>', color='gray')
    bnext.on_clicked(functools.partial(next, calibration_file=calibration_file, depthmap_dir=depthmap_dir))
    bexport_obj = Button(plt.axes([0.3, 0.0, 0.2, 0.05]), 'Export OBJ', color='gray')
    bexport_obj.on_clicked(
        functools.partial(
            export_obj, dmap))
    bexport_pcd = Button(plt.axes([0.5, 0.0, 0.2, 0.05]), 'Export PCD', color='gray')
    bexport_pcd.on_clicked(
        functools.partial(
            export_pcd, dmap))
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

    # calibration = utils.parse_calibration(calibration_file)

    # Clear export folder
    try:
        shutil.rmtree('export')
    except BaseException:
        print('no previous data to delete')
    os.mkdir('export')

    # Show viewer
    index = 0
    size = len(depth_filenames)
    show(depthmap_dir, calibration_file)
