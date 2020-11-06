import concurrent.futures
import argparse
import logging
import warnings
import matplotlib.pyplot as plt
import datetime
import os
import sys
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import azureml.core
import tensorflow.compat.v1 as tf
from PIL import Image
from get_timestamps import get_timestamps_from_rgb, get_timestamps_from_pcd
from cgm_fusion.fusion import fuse_rgbd
# from segmentation import DeepLabModel, load_model, apply_segmentation
sys.path.append('../cgm-ml')
sys.path.append(os.path.dirname(os.getcwd()))
tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)
warnings.filterwarnings("ignore")
logging.getLogger('').handlers = []
logging.basicConfig(filename='./RGBD.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s')


# function to find the closest rgbd images for a given pcd
def find_closest(rgb, pcd):

    # rgb must be sorted
    idx = rgb.searchsorted(pcd)
    idx = np.clip(idx, 1, len(rgb) - 1)
    left = rgb[idx - 1]
    right = rgb[idx]
    idx -= pcd - left < right - pcd
    return idx


def get_filename(pcd_file, rgbd_folder, qr_folder):

    # save the new data to the folder
    fused_folder, pc_filename = os.path.split(str(pcd_file))
    pcd_path_old = pcd_file

    # finding if its standing or lying artifact
    scan_type = pcd_file.split("/")[-1].split("_")[3]

    # replace the pcd and the pc_ in the path for fused data
    pc_filename = pcd_path_old.replace(".pcd", ".ply")
    pc_filename = pc_filename.replace("pc_", "pcrgb_")

    # make a folder for scan type
    qr_rgbd = os.path.join(rgbd_folder, qr_folder)
    qr_rgbd_scan = os.path.join(qr_rgbd, str(scan_type))

    # make the output rgbd filename
    rgbd_filename = pc_filename.replace(
        fused_folder.split("measure")[0], qr_rgbd_scan + "/")

    # manipulate the filename
    rgbd_filename = rgbd_filename.replace(".ply", ".pkl")
    rgbd_filename = rgbd_filename.replace("/pc/", "")

    # check if output rgbd folder exists
    rgbd_folder_ = os.path.dirname(rgbd_filename)
    if not (os.path.isfile(rgbd_folder_)):
        logging.info("Folder does not exist for " + str(rgbd_filename))
        os.makedirs(rgbd_folder_, exist_ok=True)
        logging.info("Created folder " + str(rgbd_folder_))

    ##TODO:uncomment for segmentation fusion

    # check if a segmentation for the found jpg exists
    # seg_path_ = jpg_file.replace('.jpg', '_SEG.png')
    # seg_path_,seg_file=os.path.split(seg_path_)
    # seg_folder=os.path.join(seg_folder_,qr_folder)

    # if not os.path.exists(seg_folder):
    #     os.mkdir(seg_folder)
    # seg_path=os.path.join(seg_folder,seg_file)

    # if not( os.path.exists(seg_path) ):

    #     logging.debug('applying segmentation')
    #     seg_path = apply_segmentation(image,seg_path,model)
    #     # check if the path now exists
    # if not( os.path.exists(seg_path) ):
    #         logging.error('Segmented file does not exist: ' + seg_path)

    return rgbd_filename


def process_pcd(paths, process_index=0):

    pcd_file = paths[0]
    jpg_file = paths[1]

    #rgb image
    image = Image.open(jpg_file)

    #get the qr folder
    qr_folder = str(Path(qr)).split("/")[-1]

    calibration_file = "./calibration.xml"

    if args.pickled:
        #get height and weight label for the corresponding artifact
        height = int(artifacts_file.loc[np.where(
            artifacts_file["qrcode"] == qr_folder)].iloc[0].loc["height"])
        weight = int(artifacts_file.loc[np.where(
            artifacts_file["qrcode"] == qr_folder)].iloc[0].loc["weight"])

    rgbd_filename = get_filename(pcd_file, rgbd_folder, qr_folder)

    logging.info("Going to writing new fused data to: " + rgbd_filename)

    #saving the rgbd file with labels as pickled data
    try:
        rgbdseg_arr = fuse_rgbd(calibration_file, pcd_file,
                                image)  # , seg_path)
        if args.pickled:
            labels = np.array([height, weight])
            if not labels:
                print("labels dont exist in artifacts.csv..exiting")
                sys.exit()
            data = (rgbdseg_arr, labels)
            pickle.dump(data, open(rgbd_filename, "wb"))
        else:
            #saving as a png file if not pickled data
            rgbd_filename = rgbd_filename.replace(".pkl", ".npy")
            np.save(rgbd_filename, rgbdseg_arr)

        logging.info("successfully wrote new data to" + rgbd_filename)
    except Exception as e:
        logging.error("Something went wrong.Skipping this file")

        logging.error(str(e))


def get_files(norm_rgb_time, rgb_path, norm_pcd_time, pcd_path):
    files = []
    if len(norm_rgb_time) == 0:
        print("no rgb images found")
        return []

    if len(norm_pcd_time) == 0:
        print("no pcd images found")
        return []

    for i, pcd in enumerate(norm_pcd_time):
        nn = find_closest(norm_rgb_time, pcd)

        # get the original file path
        path, filename = os.path.split(str(pcd_path[i]))

        #located the nearest jpg file of that pcd
        pcd_file = pcd_path[i]
        jpg_file = rgb_path[nn]

        files.append([pcd_file, jpg_file])

    return files


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Give in the qrcode folder to get rgbd data')
    parser.add_argument('--input', required=True,
                        metavar='inputpath',
                        type=str,
                        help='the path to the input qrcode folder')
    parser.add_argument('--output', required=True,
                        metavar='outputpath',
                        default="./output_rgbd",
                        type=str,
                        help='output rgbd folder path')
    parser.add_argument("--num_workers",
                        metavar="workers",
                        type=int,
                        default=None,
                        help="no. of cpu workers you want to process with")
    parser.add_argument(
        "--pickled",
        action='store_true',
        help="if you are processing on data of a dataset whose labels you have added into artifacts.csv")
    args = parser.parse_args()

    start = datetime.datetime.now()

    #reading artifacts.csv for pickled qrcode paths
    if args.pickled:
        artifacts = os.path.join(
            os.path.dirname(os.path.dirname(os.getcwd())),
            'data_utils/dataset_EDA/50k_pcd/artifacts.csv')
        mnt = args.input + "/qrcode/"
        artifacts_file = pd.read_csv(artifacts)
        qrcode = artifacts_file["qrcode"]
        paths = artifacts_file["storage_path"]
        unique_qr_codes = [os.path.join(mnt, line) for line in qrcode]

    else:
        folders = os.listdir(args.input)
        unique_qr_codes = [os.path.join(args.input, x) for x in folders if os.path.isdir(os.path.join(args.input, x))]

    ##validation of input directory
    if not os.path.exists(unique_qr_codes[0]):
        print("Error:invalid input paths..exiting")
        sys.exit()
    #TODO
    #loading DeepLab model
    # if args.segmented:
    #     model = load_model()

    #making output dir for storing rgbd files
    rgbd_folder = args.output
    if not os.path.exists(rgbd_folder):
        os.mkdir(rgbd_folder)

    #initialize empty lists for rgb and pcd paths of a given qr code
    rgb_paths = []
    pcd_paths = []

    #read jpg and pcd files of each qr code in the input directory.
    print("Processing..")
    start = datetime.datetime.now()
    for qr in set(unique_qr_codes):
        logging.info("reading qr code" + str(qr))
        for dirname, dirs, qr_paths in os.walk(Path(qr)):
            for file in qr_paths:
                dir_path = os.path.join(dirname, file)
                if file.endswith(".jpg"):
                    rgb_paths.append(dir_path)
                if file.endswith(".pcd"):
                    pcd_paths.append(dir_path)

    #getting the timestamps of rgb and pcd paths
        [norm_rgb_time, rgb_path] = get_timestamps_from_rgb(rgb_paths)
        [norm_pcd_time, pcd_path] = get_timestamps_from_pcd(pcd_paths)

        paths = get_files(norm_rgb_time, rgb_path, norm_pcd_time, pcd_path)

        #processing every pcd file with its nearest rgb using multiprocessing workers
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_workers) as executor:
            res = list(tqdm(executor.map(process_pcd, paths),
                            total=len(paths)))

    end = datetime.datetime.now()
    diff = end - start
    print("***Done***")
    print("total time took is {}".format(diff))
