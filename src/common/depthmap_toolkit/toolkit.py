import os
import shutil
import sys
from os import walk
from shutil import copyfile

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import depthmap
import pcd2depth


def convertAllPCDs(event):
    input = 'export'
    pcd = []
    for (dirpath, dirnames, filenames) in walk(input):
        pcd = filenames
    pcd.sort()
    try:
        shutil.rmtree('output')
    except BaseException:
        print('no previous data to delete')
    os.mkdir('output')
    os.mkdir('output/depth')
    copyfile(input + '/../camera_calibration.txt', 'output/camera_calibration.txt')
    for i in range(len(pcd)):
        pcd2depth.process(
            input + '/../camera_calibration.txt',
            input + '/' + pcd[i],
            'output/depth/' + pcd[i] + '.depth')
    print('Data exported into folder output')


def exportOBJ(event):
    depthmap.export('obj', 'output' + str(index) + '.obj')


def exportPCD(event):
    depthmap.export('pcd', 'output' + str(index) + '.pcd')


def next(event):
    plt.close()
    global index
    index = index + 1
    if (index == size):
        index = 0
    show()


def prev(event):
    plt.close()
    global index
    index = index - 1
    if (index == -1):
        index = size - 1
    show()


def swapEdges(event):
    plt.close()
    global edges
    edges = 1 - edges
    show()


def show():
    if rgb:
        depthmap.process(plt, input, depth[index], rgb[index])
    else:
        depthmap.process(plt, input, depth[index], 0)

    if edges == 1:
        depthmap.showEdges()
    else:
        depthmap.showResult()
    ax = plt.gca()
    ax.text(0.5, 1.075, depth[index], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    bprev = Button(plt.axes([0.0, 0.0, 0.1, 0.075]), '<<', color='gray')
    bprev.on_clicked(prev)
    bnext = Button(plt.axes([0.9, 0.0, 0.1, 0.075]), '>>', color='gray')
    bnext.on_clicked(next)
    bexportOBJ = Button(plt.axes([0.2, 0.0, 0.2, 0.05]), 'Export OBJ', color='gray')
    bexportOBJ.on_clicked(exportOBJ)
    bexportPCD = Button(plt.axes([0.4, 0.0, 0.2, 0.05]), 'Export PCD', color='gray')
    bexportPCD.on_clicked(exportPCD)
    bconvertPCDs = Button(plt.axes([0.6, 0.0, 0.2, 0.05]), 'Convert all PCDs', color='gray')
    bconvertPCDs.on_clicked(convertAllPCDs)
    if edges == 0:
        bshowedges = Button(plt.axes([0.0, 0.94, 0.2, 0.05]), 'Show edges', color='gray')
        bshowedges.on_clicked(swapEdges)
    else:
        bshowedges = Button(plt.axes([0.0, 0.94, 0.2, 0.05]), 'Hide edges', color='gray')
        bshowedges.on_clicked(swapEdges)
    plt.show()


#prepare
if len(sys.argv) != 2:
    print('You did not enter input folder')
    print('E.g.: python toolkit.py honor')
    sys.exit(1)

input = sys.argv[1]
depth = []
rgb = []
for (dirpath, dirnames, filenames) in walk(input + '/depth'):
    depth = filenames
for (dirpath, dirnames, filenames) in walk(input + '/rgb'):
    rgb = filenames
depth.sort()
rgb.sort()

#make sure there is a new export folder
try:
    shutil.rmtree('export')
except BaseException:
    print('no previous data to delete')
os.mkdir('export')

#show viewer
edges = 0
index = 0
size = len(depth)
show()
