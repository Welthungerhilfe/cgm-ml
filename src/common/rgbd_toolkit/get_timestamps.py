import logging
import re

import numpy as np


def get_timestamps_from_rgb(rgb_paths):
    path = [x for x in rgb_paths]
    timestamps = []

    for p in path:
        filename = p.split('/')[-1]
        value = filename.split('_')[-1]
        timestamp_value = value.replace('.jpg', '')
        timestamps.append(float(timestamp_value))

    if len(timestamps) == 0:
        error = np.array([])
        return [error, path]

    timestamps = np.asarray(timestamps)

    return [timestamps, path]


def get_timestamp_from_pcd(pcd_path):
    filename = str(pcd_path)
    infile = open(filename, 'r')
    first_line = infile.readline()

    # get the time from the header of the pcd file
    timestamp = re.findall(r"\d+\.\d+", first_line)

    # check if a timestamp is parsed from the header of the pcd file
    try:
        return_timestamp = float(timestamp[0])
    except IndexError:
        return_timestamp = []

    return return_timestamp


def get_timestamps_from_pcd(pcd_paths):
    timestamps = np.array([])
    path = [x for x in pcd_paths]

    #iterate over all paths pointing to pcds
    for p in path:
        try:
            stamp = get_timestamp_from_pcd(p)
            timestamps = np.append(timestamps, stamp)
        except IndexError:
            error = np.array([])
            logging.error("Error with timestamp in pcd")
            return [error, p]

    if len(timestamps) == 0:
        error = np.array([])
        return [error, path]
    return [timestamps, path]
