import os
import shutil
import sys
import logging
import logging.config

import pcd2depth
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.info('You did not enter pcd_dir folder and calibration file path')
        logging.info('E.g.: python convertpcd2depth.py pcd_dir calibration_file')
        sys.exit(1)
    pcd_dir = sys.argv[1]
    calibration_file = sys.argv[2]
    calibration = utils.parse_calibration(calibration_file)

    pcd = []
    for (dirpath, dirnames, filenames) in os.walk(pcd_dir):
        pcd = filenames
    pcd.sort()
    try:
        shutil.rmtree('output')
    except BaseException:
        print('no previous data to delete')
    os.makedirs('output/depth')

    width = int(240 * 0.75)
    height = int(180 * 0.75)

    for i in range(len(pcd)):
        depthmap = pcd2depth.process(calibration, pcd_dir + '/' + pcd[i], width, height)
        pcd2depth.write_depthmap('output/depth/' + pcd[i] + '.depth', depthmap, width, height)
    logging.info('Data exported into folder output')
