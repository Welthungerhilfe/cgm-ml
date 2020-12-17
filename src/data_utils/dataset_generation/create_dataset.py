import yaml
import logging
import pickle
import os
import multiprocessing
import shutil
import sys

import numpy as np
from pathlib import Path

sys.path.append('.')
sys.path.append(str(Path(__file__).parents[1]))

from data_utils import QRCodeCollector  # noqa: E402
from src.common.depthmap_toolkit import pcd2depth  # noqa: E402


if __name__ == "__main__":

    # Load the yaml file
    with open("src/data_utils/create_dataset/parameters.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Parse all the configuration variables

    DB_FILE = cfg["database"]['db_connection_file']
    NUMBER_OF_SCANS = cfg['scans']['scan_amount']
    CALIBRATION_FILE = cfg['calibration']['Lenovo']
    SCANGROUP = cfg['data']['scangroup']
    TARGET_FOLDER = cfg['paths']['target_path']
    SOURCE = cfg['paths']['source_path']
    DEBUG = cfg['debug']['debug']
    DEPTHMAP_CSV = cfg['csv_files']['depthmap_csv']
    RGB_CSV = cfg['csv_files']['rgb_csv']

    pcd_path = TARGET_FOLDER + 'pointclouds'
    if not os.path.exists(pcd_path):
        os.makedirs(pcd_path)

    depthmap_path = TARGET_FOLDER + 'depthmaps'
    if not os.path.exists(depthmap_path):
        os.makedirs(depthmap_path)

    rgb_path = TARGET_FOLDER + 'rgb'
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)

    dataset = QRCodeCollector(DB_FILE)
    logging.info("Starting the dataset Preparation:")
    data = dataset.get_all_data()
    scangroup_data = dataset.get_scangroup_data(data=data, scangroup=SCANGROUP)
    scangroup_qrcodes = dataset.get_unique_qrcode(scangroup_data)
    new_scangroup_data = dataset.get_usable_data(dataframe=scangroup_qrcodes, amount=NUMBER_OF_SCANS, scan_group=SCANGROUP)

    full_dataset = dataset.merge_qrcode_dataset(new_scangroup_data, scangroup_data)
    logging.info("Saving the csv file for EDA notebook.")
    full_dataset.to_csv(DEPTHMAP_CSV, index=False)

    # Create the RGB csv file for posenet.
    get_posenet_data = dataset.get_posenet_results()
    get_rgb_artifacts = dataset.get_artifacts()

    get_rgb_qrcodedata = dataset.merge_data_artifacts(full_dataset, get_rgb_artifacts)
    get_posenet_results = dataset.merge_data_posenet(full_dataset, get_posenet_data)

    get_posenet_results.to_csv(RGB_CSV, index=False)

    def process_depthmap_pcd(data):
        """
        Functions to process the pointclouds to depthmaps, store the pointclouds and depthmaps in
        corresponding folder structre.

        Args:
            data (rows of dataframe): A rows of dataframe having the qrcodes , storage_path, and other labels.
        """
        source_path = SOURCE + data['storage_path']
        qrcode = data['qrcode']
        pcdfile = data['storage_path'].split('/')[-1]
        depthvalues = pcd2depth.process(CALIBRATION_FILE, source_path)
        depthmaps = depthvalues[:, :, 2]
        depthmaps = np.expand_dims(depthmaps, axis=2)
        max_value = depthmaps.max()
        if max_value > 10:
            logging.warning(pcdfile)
            return
        scantype = pcdfile.split('_')[3]
        pickle_file = pcdfile.replace('.pcd', '.p')
        labels = np.array([data['height'], data['weight'], data['muac'], data['age'], data['sex'], data['tag'], data['scan_group']])
        depthmap_target_path = os.path.join(depthmap_path, qrcode)
        depthmap_complete_path = os.path.join(depthmap_target_path, scantype)
        Path(depthmap_complete_path).mkdir(parents=True, exist_ok=True)
        datapoints = (depthmaps, labels)
        depthmap_save_path = depthmap_complete_path + '/' + pickle_file
        pickle.dump(datapoints, open(depthmap_save_path, "wb"))
        pcd_target_path = os.path.join(pcd_path, qrcode)
        pcd_complete_path = os.path.join(pcd_target_path, scantype)
        Path(pcd_complete_path).mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, pcd_complete_path)

    def process_RGB(data):
        """
        Function to process the RGB images, store them in
        corresponding folder structre

        Args:
            data (rows of dataframe): A rows of dataframe having the qrcodes , storage_path, and other labels.
        """
        source_path = SOURCE + data['storage_path']
        qrcode = data['qrcode']
        imagefile = ['storage_path'].split('/')[-1]
        scantype = imagefile.split('_')[3]
        rgb_target_path = os.path.join(rgb_path, qrcode)
        rgb_complete_path = os.path.join(rgb_target_path, scantype)
        Path(rgb_complete_path).mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, rgb_complete_path)

    if DEBUG:
        for index, row in full_dataset.iterrows():
            process_depthmap_pcd(row)
    else:
        proc = multiprocessing.Pool()
        for index, row in full_dataset.iterrows():
            proc.apply_async(process_depthmap_pcd, [row])
        proc.close()
        proc.join()  # Wait for all child processes to close.

    if DEBUG:
        for index, row in get_posenet_results.iterrows():
            process_RGB(row)
    else:
        proc = multiprocessing.Pool()
        for index, row in get_posenet_results.iterrows():
            proc.apply_async(process_RGB, [row])
        proc.close()
        proc.join()  # Wait for all child processes to close.

    proc = multiprocessing.Pool()

    for index, row in get_posenet_results.iterrows():
        proc.apply_async(process_RGB, [row])  # Activate multiprocessing

    proc.close()
    proc.join()

    dataset.update_database(full_dataset, SCANGROUP)
