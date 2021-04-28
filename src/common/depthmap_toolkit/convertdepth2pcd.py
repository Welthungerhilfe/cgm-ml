import os
import shutil
import sys
import logging
import logging.config

import matplotlib.pyplot as plt

import depthmap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.info('You did not enter depthmap_dir folder')
        logging.info('E.g.: python convertdepth2pcd.py depthmap_dir')
        sys.exit(1)

    depthmap_dir = sys.argv[1]
    depth = []
    for (dirpath, dirnames, filenames) in os.walk(depthmap_dir + '/depth'):
        depth = filenames
    depth.sort()
    try:
        shutil.rmtree('export')
    except BaseException:
        print('no previous data to delete')
    os.mkdir('export')
    for i in range(len(depth)):
        depthmap.process(plt, depthmap_dir, depth[i], 0)
        depthmap.export('pcd', 'output' + depth[i] + '.pcd')

    logging.info('Data exported into folder export')
