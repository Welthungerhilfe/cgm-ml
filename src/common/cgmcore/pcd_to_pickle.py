import pickle
import pandas as pd
import numpy as np
from pyntcloud import PyntCloud
import pathlib
import os
import logging
import logging.config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


source_path = '/mnt/depthmap/pcd_50k/'
target_path = '/mnt/depthmap/pcd_50k/pcd_50k_pkl'
dataset = '../../gitrepo/cgm-ml-service/dataset_EDA/opensource_pcd_data/artifacts.csv'

if not os.path.exists(target_path):
    os.mkdir(target_path)


def load_pcd_as_ndarray(pcd_path):
    """
    Loads a PCD-file. Yields a numpy-array.
    """
    pointcloud = PyntCloud.from_file(pcd_path)
    values = pointcloud.points.values
    return values


def pcd_to_pickle(dataset):
    """
    --dataset: A csv file conataining height, weight, qrcode, storage path for pcd

    return:
    -- folder with pickle file having pointcloud, height and weight
    """
    data = pd.read_csv(dataset)
    for index, rows in data.iterrows():
        pcd_path = source_path + rows['storage_path']
        pointcloud = load_pcd_as_ndarray(pcd_path)
        label = np.array([float(rows['height']), float(rows['weight'])])
        qrcode_path = os.path.join(target_path, rows['qrcode'])
        pickle_filename = os.path.basename(rows['artifacts']).replace(".pcd", ".p")
        pathlib.Path(qrcode_path).mkdir(parents=True, exist_ok=True)
        pickle_output_path = os.path.join(qrcode_path, pickle_filename)
        pickle.dump((pointcloud, label), open(pickle_output_path, "wb"))


def main():
    logging.info("Starting the data preparation.")
    pcd_to_pickle(dataset)
    logging.info("dataset preparation finished.Check the %s path for outputs", target_path)


if __name__ == "__main__":
    main()
