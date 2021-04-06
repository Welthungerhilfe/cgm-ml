import zipfile
import pickle
import multiprocessing
import json
import logging

import numpy as np
import cv2 as cv
from glob2 import glob
from pathlib import Path
from glob import glob1

TARGET_HEIGHT = 180
TARGET_WIDTH = 240
TARGET_PATH = '/mnt/huawei_dataset/anon-rgbd-5kscans'
SOURCE_PATH = '/mnt/huawei_dataset/huawei_data'


def load_depth(filename):
    with zipfile.ZipFile(filename) as z:
        with z.open('data') as f:
            line = str(f.readline())[2:-3]
            header = line.split("_")
            res = header[0].split("x")
            # print(res)
            width = int(res[0])
            height = int(res[1])
            depthScale = float(header[1])
            max_confidence = float(header[2])
            data = f.read()
            f.close()
        z.close()
    return data, width, height, depthScale, max_confidence


def prepare_depthmap(data, width, height, depthScale):
    # prepare array for output
    output = np.zeros((width, height, 1))
    for cx in range(width):
        for cy in range(height):
            #             output[cx][height - cy - 1][0] = parse_confidence(cx, cy)
            #             output[cx][height - cy - 1][1] = im_array[cy][cx][1] / 255.0 #test matching on RGB data
            #             output[cx][height - cy - 1][2] = 1.0 - min(parse_depth(cx, cy) / 2.0, 1.0) #depth data scaled to be visible
            # depth data scaled to be visible
            output[cx][height - cy - 1] = parse_depth(cx, cy, data, depthScale)
    return (
        np.array(
            output,
            dtype='float32').reshape(
            width,
            height),
        height,
        width)


def parse_depth(tx, ty, data, depthScale):
    depth = data[(int(ty) * TARGET_WIDTH + int(tx)) * 3 + 0] << 8
    depth += data[(int(ty) * TARGET_WIDTH + int(tx)) * 3 + 1]
    depth *= depthScale
    return depth


def check_correspondence(depth_frame, rgb_file_list):
    depth_name = depth_frame.split('.depth')[0]
    frame_name = depth_name.split('depth_')
    rgb_frame = f'rgb_{frame_name[-1]}.jpg'
    assert rgb_frame in rgb_file_list
    return depth_frame, rgb_frame


def image_resize(image_path):
    image = cv.imread(image_path)
    image_rotated = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    image_rotated = cv.resize(image_rotated, (TARGET_HEIGHT, TARGET_WIDTH))
    return image_rotated


def read_json(filepath):
    with open(filepath) as json_data:
        label_data = json.load(json_data)
    return label_data


def process_depthmap(depthmaps):
    split_dirpath = depthmaps.split('/depth')[0]
    json_fpath = f'{split_dirpath}/targets.json'
    label_data = read_json(json_fpath)
    labels = np.array([label_data['height'], label_data['weight'],
                       label_data['muac'], label_data['age'], label_data['sex']])
    qrcode = depthmaps.split('/')[5]
    qrcode_dirpath = f'{TARGET_PATH}/{qrcode}'
    Path(qrcode_dirpath).mkdir(parents=True, exist_ok=True)
    rgb_dirpath = f'{split_dirpath}/rgb'
    rgb_list = glob1(rgb_dirpath, '*.jpg')
    abs_depth_pattern = f'{depthmaps}/*.depth'
    depthmap_files = glob(abs_depth_pattern)
    for unique_depthmaps in depthmap_files:
        try:
            depthmap_image_path, image_path = check_correspondence(unique_depthmaps, rgb_list)
        except Exception as e:
            message = f" Error '{e.message}' occurred. Depthmap file '{depthmap_files}' not found."
            logging.info(message)
            continue
        scan_type = image_path.split('_')[3]
        artifact_name = image_path.split('rgb_')[1]
        scan_type_dirpath = f'{qrcode_dirpath}/{scan_type}'
        pickle_file = artifact_name.replace('.jpg', '.p')
        full_fpath = f'{scan_type_dirpath}/{pickle_file}'
        Path(scan_type_dirpath).mkdir(parents=True, exist_ok=True)
        data, width, height, depthScale, max_confidence = load_depth(depthmap_image_path)
        depthmap_huawei = prepare_depthmap(data, width, height, depthScale)
        image_full_fpath = f'{rgb_dirpath}/{image_path}'
        resized_image = image_resize(image_full_fpath)
        pickled_data = (resized_image, depthmap_huawei[0], labels)
        pickle.dump(pickled_data, open(full_fpath, "wb"))


if __name__ == "__main__":
    source_pattern = f'{SOURCE_PATH}'
    dataset_list = []
    proc = multiprocessing.Pool()
    for elem in Path(source_pattern).rglob('*/depth'):
        proc.apply_async(dataset_list.append(elem))
    proc.close()
    proc.join()  # Wait for all child processes to close
    proc = multiprocessing.Pool()
    for depthimages in dataset_list:
        # process_depthmap(depthimages)
        # launch a process for each file (ish).
        # The result will be approximately one process per CPU core available.
        proc.apply_async(process_depthmap, [depthimages])

    proc.close()
    proc.join()  # Wait for all child processes to close
