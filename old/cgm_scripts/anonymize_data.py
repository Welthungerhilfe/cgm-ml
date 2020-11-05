#
# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Tristan Behrens <tristan@ai-guru.de> for Welthungerhilfe
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import os
import sys
sys.path.insert(0, "..")
import numpy as np
from cgmcore import utils
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import shutil
import cv2
import face_recognition
import glob
from tqdm import tqdm
import random
import logging

# Create logger.
logger = logging.getLogger("anonymize_data.py")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("anonymize_data.log")
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Modes the program can run in.
modes = ["file", "scan", "all"]

def main():

    # Print usage.
    if len(sys.argv) != 4:
        print("")
        print("Usage: python anonymize_data.py MODE INPUT OUTPUT")
        print("  MODE: file|scan|all")
        print("  INPUT: Path to specific file, or a specific scan, or all scans.")
        print("  OUTPUT: Path for the anonymized data.")
        print("")
        print("Examples:")
        print("  anonymize_data.py file /localssd/qrcode/MH_WHH_0010/measurements/1537166990387/rgb/rgb_MH_WHH_0010_1537166990387_110_1750.069971052.jpg /localssd/anondata/")
        print("  anonymize_data.py file /localssd/qrcode/MH_WHH_0010 /localssd/anondata/")
        exit(0)

    # Process the command line arguments.
    mode = sys.argv[1]
    if mode not in modes:
        logger.error("Invalid mode {}".format(mode))
        exit(0)
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    # Process all files.
    if mode == "all":
        results = process_all(input_path, output_path)

    # Process a scan.
    elif mode == "scan":
        results = process_scan(input_path, output_path)
        for result in results:
            logger.info(" ".join(result))

    # Process a single file.
    elif mode == "file":
        result = process_file(input_path, output_path)
        results = [result]
        for result in results:
            logger.info(" ".join(result))



def process_all(all_path, output_path):

    # Getting the paths of the qr-codes.
    logger.info("Gathering all qr codes...")
    qrcode_paths = glob.glob(os.path.join(all_path, "*"))
    qrcode_paths = [path for path in qrcode_paths if os.path.isdir(path) == True]
    qrcode_paths = [path for path in qrcode_paths if os.path.exists(os.path.join(path, "measurements"))]
    qrcode_paths = sorted(qrcode_paths)

    # Do a quality check on the qr code paths.
    if len(qrcode_paths) == 0:
        logger.info("No measurements found at \"{}\"!".format(all_path))

    # This method is called in multiple processes.
    def process_qrcode_path(qrcode_path):

        results_all = []
        scan_paths = glob.glob(os.path.join(qrcode_path, "measurements", "*"))
        scan_paths = [path for path in scan_paths if os.path.isdir(path) == True]
        for scan_path in scan_paths:
            results = process_scan(scan_path, output_path)
            #results_all.extend(results)
            for result in results:
                logger.info(" ".join(result))

        return results_all

    # Run this in multiprocess mode.
    results = utils.multiprocess(
        qrcode_paths,
        process_method=process_qrcode_path,
        process_individial_entries=True,
        progressbar=False,
        number_of_workers=10,
        disable_gpu=True
    )
    return []


def process_scan(scan_path, output_path, show_progress=True):
    """
    Processes a single scan.
    """

    # Check if we have a folder.
    if os.path.isdir(scan_path) == False:
        logger.error("Must provide a folder!")
        raise Exception()

    # See if we are really in a scan.
    paths = glob.glob(os.path.join(scan_path, "*"))
    if os.path.join(scan_path, "rgb") not in paths and os.path.join(scan_path, "pc") not in paths:
        logger.error("Direct subfolders rgb or pc not found in input folder!")
        raise Exception()

    # Walk the scan path.
    walker = os.walk(scan_path)
    if show_progress == True:
        walker = tqdm(walker)
    results = []
    for root_path, _, filenames in walker:

        # Skip everything else.
        if root_path.endswith("rgb") == False and root_path.endswith("pc") == False:
            continue

        # Go through all filenames.
        for filename in filenames:
            file_path = os.path.join(root_path, filename)
            result = process_file(file_path, output_path)
            results.append(result)

    return results


def process_file(file_path, output_path):
    """
    Processes a single file.
    """

    # Must be a file.
    if os.path.isfile(file_path) == False:
        logger.error("Must provide a file!")
        raise Exception()

    # Must be an image.
    if (file_path.endswith("jpg") or file_path.endswith("png") or file_path.endswith("pcd") or file_path.endswith("ply") or file_path.endswith("vtk")) == False:
        logger.error("File extension of \"{}\" is not valid! Allowed: jpg/png/pcd/ply/vtk".format(file_path))
        raise Exception()

    # Check if the folder structure is correct.
    file_path_split = file_path.split("/")
    if file_path_split[-6] != "qrcode":
        logger.error("Expected \"qrcode\" in path, got \"{}\"".format(file_path_split[-6]))
        raise Exception()
    if file_path_split[-2] != "rgb" and file_path_split[-2] != "pc":
        logger.error("Expected \"rgb\" or \"pc\" in path, got \"{}\"".format(file_path_split[-2]))
        raise Exception()

    # Make sure the output folder exists.
    file_output_folder = os.path.join(output_path, *file_path_split[-5:-1])
    if os.path.exists(file_output_folder) == False:
        os.makedirs(file_output_folder)

    # This is the output file name.
    file_output_path = os.path.join(output_path, *file_path_split[-5:])

    # File already exists.
    if os.path.exists(file_output_path) == True:
        return (file_path, "skipped because already exists")

    # Pointclouds are just copied.
    if file_path.endswith(".pcd"):
        shutil.copy(file_path, file_output_path)
        return (file_path, "copied")
    elif file_path.endswith(".jpg") or file_path.endswith(".png"):
        result = blur_faces_in_file(file_path, file_output_path)
        return (file_path, result)






if __name__ == '__main__':
    main()
