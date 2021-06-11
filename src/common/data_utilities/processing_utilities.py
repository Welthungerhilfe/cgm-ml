from typing import Tuple
from pathlib import Path
import pickle
import zipfile

import numpy as np
from skimage.transform import resize

WIDTH = 240
HEIGHT = 180
NORMALIZATION_VALUE = 7.5
IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH = HEIGHT, WIDTH


def load_depth(fpath: str) -> Tuple[bytes, int, int, float, float]:
    """Take ZIP file and extract depth and metadata
    Args:
        fpath (str): File path to the ZIP
    Returns:
        depth_data (bytes): depthmap data
        width(int): depthmap width in pixel
        height(int): depthmap height in pixel
        depth_scale(float)
        max_confidence(float)
    """

    with zipfile.ZipFile(fpath) as z:
        with z.open('data') as f:
            # Example for a first_line: '180x135_0.001_7_0.57045287_-0.0057296_0.0022602521_0.82130724_-0.059177425_0.0024800065_0.030834956'
            first_line = f.readline().decode().strip()

            file_header = first_line.split("_")

            # header[0] example: 180x135
            width, height = file_header[0].split("x")
            width, height = int(width), int(height)
            depth_scale = float(file_header[1])
            max_confidence = float(file_header[2])

            depth_data = f.read()
    return depth_data, width, height, depth_scale, max_confidence


def parse_depth(tx: int, ty: int, data: bytes, depth_scale: float, width: int) -> float:
    assert isinstance(tx, int)
    assert isinstance(ty, int)

    depth = data[(ty * width + tx) * 3 + 0] << 8
    depth += data[(ty * width + tx) * 3 + 1]

    depth *= depth_scale
    return depth


def preprocess_depthmap(depthmap):
    return depthmap.astype("float32")


def preprocess(depthmap):
    depthmap = preprocess_depthmap(depthmap)
    depthmap = depthmap / NORMALIZATION_VALUE
    depthmap = resize(depthmap, (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
    depthmap = depthmap.reshape((depthmap.shape[0], depthmap.shape[1], 1))
    return depthmap


def prepare_depthmap(data: bytes, width: int, height: int, depth_scale: float) -> np.array:
    """Convert bytes array into np.array"""
    output = np.zeros((width, height, 1))
    for cx in range(width):
        for cy in range(height):
            # depth data scaled to be visible
            output[cx][height - cy - 1] = parse_depth(cx, cy, data, depth_scale, width)
    arr = np.array(output, dtype='float32')
    return arr.reshape(width, height)


def get_depthmaps(fpaths):
    depthmaps = []
    for fpath in fpaths:
        data, width, height, depthScale, _ = load_depth(fpath)
        depthmap = prepare_depthmap(data, width, height, depthScale)
        depthmap = preprocess(depthmap)
        depthmaps.append(depthmap)

    depthmaps = np.array(depthmaps)
    return depthmaps


def create_and_save_pickle(zip_input_full_path, timestamp, scan_id, scan_step, target_tuple, order_number):
    """Side effect: Saves and returns file path"""
    depthmaps = get_depthmaps([zip_input_full_path])
    # if DEBUG:
    #     print(depthmaps.shape, depthmaps[0,0,0,0])

    pickle_output_path = f"qrcode/{scan_id}/{scan_step}/pc_{scan_id}_{timestamp}_{scan_step}_{order_number}.p"  # '/tmp/abc.p'
    pickle_output_full_path = f"/dbfs{DBFS_DIR}/{pickle_output_path}"
    Path(pickle_output_full_path).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump((depthmaps, np.array(target_tuple)), open(pickle_output_full_path, "wb"))
    return pickle_output_full_path

def process_artifact_tuple(artifact_tuple):
    """Side effect: Saves and returns file path"""
    artifact_dict = {idx2col[i]: el for i, el in enumerate(artifact_tuple)}
    # if DEBUG:
    #     print('artifact_dict', artifact_dict)
    target_tuple = (artifact_dict['height'], artifact_dict['weight'], artifact_dict['muac'])
    zip_input_full_path = f"/dbfs{DBFS_DIR}/{artifact_dict['file_path']}"

    pickle_output_full_path = create_and_save_pickle(
        zip_input_full_path=zip_input_full_path,
        timestamp=artifact_dict['timestamp'],
        scan_id=artifact_dict['scan_id'],
        scan_step=artifact_dict['scan_step'],
        target_tuple=target_tuple,
        order_number=artifact_dict['order_number'],
    )
    return pickle_output_full_path