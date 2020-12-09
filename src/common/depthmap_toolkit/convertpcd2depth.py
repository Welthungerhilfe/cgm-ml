import os
import shutil
import sys
from shutil import copyfile

import pcd2depth

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('You did not enter pcd_dir folder')
        print('E.g.: python convertpcd2depth.py pcd_dir')
        sys.exit(1)
    pcd_dir = sys.argv[1]
    pcd = []
    for (dirpath, dirnames, filenames) in os.walk(pcd_dir):
        pcd = filenames
    pcd.sort()
    try:
        shutil.rmtree('output')
    except BaseException:
        print('no previous data to delete')
    os.mkdir('output')
    os.mkdir('output/depth')
    copyfile(pcd_dir + '/../camera_calibration.txt', 'output/camera_calibration.txt')
    for i in range(len(pcd)):
        depthmap = pcd2depth.process('camera_calibration.txt', pcd_dir + '/' + pcd[i])
        pcd2depth.write_depthmap('output/depth/' + pcd[i] + '.depth', depthmap)
    print('Data exported into folder output')
