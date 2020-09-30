import warnings
warnings.filterwarnings("ignore")
import unittest
from cgmcore.datagenerator import DataGenerator, get_dataset_path, create_datagenerator_from_parameters
import os
import glob

class TestEtl(unittest.TestCase):

    #@unittest.skip("demonstrating skipping")
    def test_pointcloud_generation(self):

        # Get the dataset path.
        dataset_path = get_dataset_path()
        assert os.path.exists(dataset_path), "Dataset-path \"{}\" does not exist! Did you specify a path in datasetpath.txt?".format(dataset_path)
        print("Dataset path:", dataset_path)

        # Get the QR-Codes.
        qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
        assert len(qrcode_paths) > 0, "No QR-codes at dataset-path \"{}\"!".format(dataset_path)
        print("QR-Code paths:", qrcode_paths)

        # Only folders allowed!
        for qrcode_path in qrcode_paths:
            assert os.path.isdir(qrcode_path), "Unexpected file \"{}\". Only folders allowed!".format(qrcode_path)

        # Check if there are folders for manual measurements.
        manual_measurements_paths_all = []
        for qrcode_path in qrcode_paths:
            manual_measurements_paths = glob.glob(os.path.join(qrcode_path, "*"))

            # Each QR-code must have at least one manual measurement.
            assert len(manual_measurements_paths) > 0, "No manual measurements at qrcode-path \"{}\"!".format(qrcode_path)

            # Only folders allowed.
            for manual_measurements_path in manual_measurements_paths:
                assert os.path.isdir(manual_measurements_path), "Unexpected file \"{}\". Only folders allowed!".format(manual_measurements_path)
            manual_measurements_paths_all.extend(manual_measurements_paths)
        print("Manual measurements paths:", manual_measurements_paths_all)

        # Check individual manual measurements.
        for manual_measurements_path in manual_measurements_paths_all:
            paths =  glob.glob(os.path.join(manual_measurements_path, "*"))
            assert len(manual_measurements_paths) > 0, "Path \"{}\" not expected to be empty!".format(manual_measurements_path)

            # Forbid folders.
            for path in paths:
                assert os.path.isdir(path) == False, "Unexpected folder \"{}\". Only folders allowed!".format(manual_measurements_path)

            targets_path = os.path.join(manual_measurements_path, "targets.json")
            assert os.path.exists(targets_path), "targets.json does not exist!"
            print("Targets-file path:", targets_path)

            # TODO check for pointclouds too!
