import os
import shutil
import sys
from os import walk

import matplotlib.pyplot as plt

import depthmap

if len(sys.argv) != 2:
    print('You did not enter input folder')
    print('E.g.: python convertdepth2pcd.py samsung')
    sys.exit(1)

input = sys.argv[1]
depth = []
for (dirpath, dirnames, filenames) in walk(input + '/depth'):
    depth = filenames
depth.sort()
try:
    shutil.rmtree('export')
except BaseException:
    print('no previous data to delete')
os.mkdir('export')
for i in range(len(depth)):
    depthmap.process(plt, input, depth[i], 0)
    depthmap.export('pcd', 'output' + depth[i] + '.pcd')

print('Data exported into folder export')
