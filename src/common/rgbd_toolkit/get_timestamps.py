import numpy as np


def get_timestamps_from_rgb(rgb_paths):

    #timestamps = [x[2] for x in all_rgb]
    path = [x for x in rgb_paths]

    timestamps = []

    for p in path:

        filename = p.split('/')[-1]
        value = filename.split('_')[-1]
        timestamp_value = value.replace('.jpg', '')
        timestamps.append(float(timestamp_value))

    if(len(timestamps) == 0):
        error = np.array([])
        return [error, path]

    timestamps = np.asarray(timestamps)
    # print("timestamp rgb",timestamps)
    # print("rgb path",path)
    return [timestamps, path]


def get_timestamp_from_pcd(pcd_path):
    filename = str(pcd_path)
    infile = open(filename, 'r')
    firstLine = infile.readline()

    # get the time from the header of the pcd file
    import re
    timestamp = re.findall(r"\d+\.\d+", firstLine)

    # check if a timestamp is parsed from the header of the pcd file
    try:
        return_timestamp = float(timestamp[0])
    except IndexError:
        return_timestamp = []

    return return_timestamp  # index error? IndexError


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

    #print("path",path)

    if(len(timestamps) == 0):
        error = np.array([])
        return [error, path]
    # print("timestamp pcd",timestamps)
    # print("pcd path",path)
    return [timestamps, path]
