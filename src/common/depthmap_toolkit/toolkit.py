import os
import shutil
import sys
from os import walk
import logging
import logging.config
from pathlib import Path
import functools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from depthmap import Depthmap
from exporter import export_obj, export_pcd
from visualisation import render_plot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

# click on data
LAST = [0, 0, 0]
# INDEX of the current depthmap/rgb frame
INDEX = 0
# current depthmap
DMAP = 0


def onclick(event):
    global DMAP
    global LAST
    if event.xdata is not None and event.ydata is not None:
        x = int(event.ydata)
        y = DMAP.height - int(event.xdata) - 1
        if x > 1 and y > 1 and x < DMAP.width - 2 and y < DMAP.height - 2:
            depth = DMAP.parse_depth(x, y)
            if depth:
                res = DMAP.convert_2d_to_3d(1, x, y, depth)
                if res:
                    diff = [LAST[0] - res[0], LAST[1] - res[1], LAST[2] - res[2]]
                    dst = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                    res.append(dst)
                    logging.info('x=%s, y=%s, depth=%s, diff=%s', str(res[0]), str(res[1]), str(res[2]), str(res[3]))
                    LAST[0] = res[0]
                    LAST[1] = res[1]
                    LAST[2] = res[2]
                    return
            logging.info('no valid data')


def export_object(event):
    global DMAP
    fname = f'output{INDEX}.obj'
    export_obj('export/' + fname, DMAP, triangulate=True)


def export_pointcloud(event):
    global DMAP
    fname = f'output{INDEX}.pcd'
    export_pcd('export/' + fname, DMAP)


def next_click(event, calibration_file: str, depthmap_dir: str):
    global INDEX
    INDEX = INDEX + 1
    if (INDEX == size):
        INDEX = 0
    show(depthmap_dir, calibration_file)


def prev_click(event, calibration_file: str, depthmap_dir: str):
    global INDEX
    INDEX = INDEX - 1
    if (INDEX == -1):
        INDEX = size - 1
    show(depthmap_dir, calibration_file)


def show(depthmap_dir: str, calibration_file: str):
    global DMAP
    fig.canvas.manager.set_window_title(depth_filenames[INDEX])
    rgb_filename = rgb_filenames[INDEX] if rgb_filenames else 0
    DMAP = Depthmap.create_from_file(depthmap_dir, depth_filenames[INDEX], rgb_filename, calibration_file)

    angle = DMAP.get_angle_between_camera_and_floor()
    logging.info('angle between camera and floor is %f', angle)

    render_plot(DMAP)
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

    # Clear export folder
    try:
        shutil.rmtree('export')
    except BaseException:
        print('no previous data to delete')
    os.mkdir('export')

    # Show viewer
    INDEX = 0
    size = len(depth_filenames)
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', functools.partial(onclick))
    bprev = Button(plt.axes([0.0, 0.0, 0.1, 0.075]), '<<', color='gray')
    bprev.on_clicked(functools.partial(prev_click, calibration_file=calibration_file, depthmap_dir=depthmap_dir))
    bnext = Button(plt.axes([0.9, 0.0, 0.1, 0.075]), '>>', color='gray')
    bnext.on_clicked(functools.partial(next_click, calibration_file=calibration_file, depthmap_dir=depthmap_dir))
    bexport_obj = Button(plt.axes([0.3, 0.0, 0.2, 0.05]), 'Export OBJ', color='gray')
    bexport_obj.on_clicked(functools.partial(export_object))
    bexport_pcd = Button(plt.axes([0.5, 0.0, 0.2, 0.05]), 'Export PCD', color='gray')
    bexport_pcd.on_clicked(functools.partial(export_pointcloud))
    background = Button(plt.axes([0.0, 0.0, 1.0, 1.0]), '', color='white')
    show(depthmap_dir, calibration_file)
