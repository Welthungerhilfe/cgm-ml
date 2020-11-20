import os
import shutil
import sys
from os import walk
from shutil import copyfile

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import depthmap
import pcd2depth


DEPTHMAP_DIR = None


def convert_all_pcds(event):
    input_dir = 'export'
    pcd = []
    for _, _, filenames in walk(input_dir):
        pcd = filenames
    pcd.sort()
    try:
        shutil.rmtree('output')
    except BaseException:
        print('no previous data to delete')
    os.mkdir('output')
    os.mkdir('output/depth')
    copyfile(input_dir + '/../camera_calibration.txt', 'output/camera_calibration.txt')
    for i in range(len(pcd)):
        depthmap = pcd2depth.process(input_dir + '/../camera_calibration.txt', input_dir + '/' + pcd[i])
        pcd2depth.write_depthmap('output/depth/' + pcd[i] + '.depth', depthmap)
    print('Data exported into folder output')


def export_obj(event):
    depthmap.export('obj', 'output' + str(index) + '.obj')


def export_pcd(event):
    depthmap.export('pcd', 'output' + str(index) + '.pcd')


def next(event):
    plt.close()
    global index
    index = index + 1
    if (index == size):
        index = 0
    show(DEPTHMAP_DIR)


def prev(event):
    plt.close()
    global index
    index = index - 1
    if (index == -1):
        index = size - 1
    show(DEPTHMAP_DIR)


def show(depthmap_dir):
    if rgb:
        depthmap.process(plt, depthmap_dir, depth[index], rgb[index])
    else:
        depthmap.process(plt, depthmap_dir, depth[index], 0)

    depthmap.show_result()
    ax = plt.gca()
    ax.text(0.5, 1.075, depth[index], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    bprev = Button(plt.axes([0.0, 0.0, 0.1, 0.075]), '<<', color='gray')
    bprev.on_clicked(prev)
    bnext = Button(plt.axes([0.9, 0.0, 0.1, 0.075]), '>>', color='gray')
    bnext.on_clicked(next)
    bexport_obj = Button(plt.axes([0.2, 0.0, 0.2, 0.05]), 'Export OBJ', color='gray')
    bexport_obj.on_clicked(export_obj)
    bexport_pcd = Button(plt.axes([0.4, 0.0, 0.2, 0.05]), 'Export PCD', color='gray')
    bexport_pcd.on_clicked(export_pcd)
    bconvertPCDs = Button(plt.axes([0.6, 0.0, 0.2, 0.05]), 'Convert all PCDs', color='gray')
    bconvertPCDs.on_clicked(convert_all_pcds)
    plt.show()


if __name__ == "__main__":
    # Prepare
    if len(sys.argv) != 2:
        print('You did not enter depthmap_dir folder')
        print('E.g.: python toolkit.py depthmap_dir')
        sys.exit(1)

    depthmap_dir = sys.argv[1]
    DEPTHMAP_DIR = depthmap_dir
    depth = []
    rgb = []
    for (dirpath, dirnames, filenames) in walk(depthmap_dir + '/depth'):
        depth = filenames
    for (dirpath, dirnames, filenames) in walk(depthmap_dir + '/rgb'):
        rgb = filenames
    depth.sort()
    rgb.sort()

    # Make sure there is a new export folder
    try:
        shutil.rmtree('export')
    except BaseException:
        print('no previous data to delete')
    os.mkdir('export')

    # Show viewer
    index = 0
    size = len(depth)
    show(depthmap_dir)
