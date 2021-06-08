import os
import shutil
import sys
import logging
import logging.config

import depthmap
import utils

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.info('You did not enter depthmap_dir folder and calibration file path')
        logging.info('E.g.: python convertdepth2pcd.py depthmap_dir calibration file')
        sys.exit(1)

    depthmap_dir = sys.argv[1]
    calibration_file = sys.argv[2]

    calibration = utils.parse_calibration(calibration_file)

    depth_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(depthmap_dir + '/depth'):
        depth_filenames.extend(filenames)
    depth_filenames.sort()
    try:
        shutil.rmtree('export')
    except BaseException:
        print('no previous data to delete')
    os.mkdir('export')
    for filename in depth_filenames:
        width, height, depth_scale, max_confidence, data, matrix = depthmap.process(depthmap_dir, filename, 0)
        output_filename = f'output{filename}.pcd'
        depthmap.export('pcd', output_filename, width, height, data, depth_scale, calibration, max_confidence, matrix)

    logging.info('Data exported into folder export')
