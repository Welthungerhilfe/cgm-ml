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
modes = ["scan", "all"]


def main():

    # Print usage.
    if len(sys.argv) != 4:
        print("")
        print("Usage: python create_gallery.py MODE INPUT OUTPUT")
        print("  MODE: scan|all")
        print("  INPUT: Path to a specific scan, or to all scans.")
        print("  OUTPUT: Path for the anonymized data.")
        print("")
        print("Examples:")
        print("  create_gallery.py all /localssd/qrcode /localssd/anondata/")
        print("  create_gallery.py file /localssd/qrcode/MH_WHH_0010 /localssd/anondata/")
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
            process_scan(scan_path, output_path)

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
        if root_path.endswith("rgb") == False:
            continue

        # Create gallery.
        create_galleries(root_path, filenames, output_path)

    return results



def create_galleries(dirpath, filenames, output_path):
    dirpath_split = dirpath.split("/")
    timestamp = dirpath_split[-2]
    qrcode = dirpath_split[-4]

    # Filter files by poses. Create one gallery per qrcode, per timestamp, and per pose.
    infixes = ["_100_", "_200_", "_104_", "_101_", "_201_", "_107_", "_102_", "_202_", "_110_"]
    for infix in infixes:
        filtered_filenames = [os.path.join(dirpath, filename) for filename in filenames if infix in filename]
        if len(filtered_filenames) == 0:
            continue

        # Create the gallery.
        gallery_file_name = "{}_{}_{}.jpg".format(qrcode, timestamp, infix.replace("_", ""))
        gallery_file_path = os.path.join(output_path, gallery_file_name)
        utils.render_artifacts_as_gallery(
            filtered_filenames,
            targets=None,
            qr_code=qrcode,
            timestamp=timestamp,
            num_columns=10,
            target_size=(1920 // 8, 1080 // 8),
            image_path=gallery_file_path,
            use_plt=False
        )
        logger.info("Wrote gallery to {}".format(gallery_file_path))


if __name__ == "__main__":
    main()
