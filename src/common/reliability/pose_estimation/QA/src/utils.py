import os
import glob


def get_file_name(im_path):
    filename_arr = im_path.split("/scans/")
    filename = filename_arr[1]
    return filename


def get_rgb_files(paths):
    # Prepare the list of all the rgb files in dataset
    rgb_paths = []
    for path in paths:
        rgb_paths.extend(glob.glob(os.path.join(path, "**", "*.jpg")))
    return rgb_paths


def get_column_list(rgb_path_list):
    # Prepare the list of all artifact with its corresponding scan type,
    # qrcode, target and prediction

    qrcode_list, artifact_list = [], []

    for idx, path in enumerate(rgb_path_list):
        sub_folder_list = path.split('/')
        qrcode_list.append(sub_folder_list[-3])
        artifact_list.append(sub_folder_list[-1])

    return qrcode_list, artifact_list
