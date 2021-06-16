import os
import shutil

import sys
import logging
import logging.config

import pcd2depth
from depthmap import parse_calibration

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.info('You did not enter pcd_dir folder and calibration file path')
        logging.info('E.g.: python convertpcd2depth.py pcd_dir calibration_file')
        sys.exit(1)

    pcd_dir = sys.argv[1]
    calibration_file = sys.argv[2]

    calibration = parse_calibration(calibration_file)

    depth_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(pcd_dir):
        depth_filenames.extend(filenames)
    depth_filenames.sort()
    try:
        shutil.rmtree('output')
    except BaseException:
        print('no previous data to delete')
    os.makedirs('output/depth')

    # works for lenovo
    width = int(240 * 0.75)
    height = int(180 * 0.75)

    for filename in depth_filenames:
        input_filename = f'{pcd_dir}/{filename}'
        depthmap = pcd2depth.process(calibration, input_filename, width, height)
        output_filename = f'output/depth/{filename}.depth'
        pcd2depth.write_depthmap(output_filename, depthmap, width, height)
    logging.info('Data exported into folder output')
