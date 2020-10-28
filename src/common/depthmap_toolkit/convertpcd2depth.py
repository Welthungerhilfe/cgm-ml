import os
import shutil
import sys
from os import walk
from shutil import copyfile

import pcd2depth

if len(sys.argv) != 2:
    print('You did not enter input folder')
    print('E.g.: python convertpcd2depth.py pcd')
    sys.exit(1)

input = sys.argv[1]
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
    pcd2depth.process('camera_calibration.txt', input + '/' + pcd[i], 'output/depth/' + pcd[i] + '.depth')
print('Data exported into folder output')
