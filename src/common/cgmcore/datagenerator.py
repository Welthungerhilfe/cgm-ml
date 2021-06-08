'''
This is a data-generator that works on the raw data. The raw data is the data that is created by the mobile app.

Note: This file is only here for historical reasons. And it might be gone anytime.
'''

from __future__ import absolute_import
import os
import logging
import logging.config
import numpy as np
import glob2
import json
import random
import keras.preprocessing.image as image_preprocessing
import progressbar
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import multiprocessing as mp
import uuid
import pickle
from . import utils

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


class DataGenerator(object):
    """
    This class generates data for training.
    """

    def __init__(
        self,
        dataset_path,
        input_type,
        output_targets,
        sequence_length=0,
        image_target_shape=(160, 90),
        voxelgrid_target_shape=(32, 32, 32),
        voxel_size_meters=0.01,
        voxelgrid_random_rotation=False,
        pointcloud_target_size=32000,
        pointcloud_random_rotation=False
    ):
        """
        Initializes a DataGenerator.

        Args:
            dataset_path (string): Where the raw data is.
            input_type (string): Specifies how the input-data for the Neural Network looks like.
                                 Either 'image', 'pointcloud', 'voxgrid'.
            output_targets (list of strings): A list of targets for the Neural Network.
                                              For example *['height', 'weight']*.
            sequence_length (int): Specifies the lenght of the sequences. 0 would yield no sequence at all.
            image_target_shape (2D tuple of ints): Target shape of the images.
            voxelgrid_target_shape (3D tuple of ints): Target shape of the voxelgrids.
            voxel_size_meters (float): Size of the voxels. That is, edge length.
            voxelgrid_random_rotation (bool): If True voxelgrids will be rotated randomly.
            pointcloud_target_size (int): Target size of the pointclouds.
            pointcloud_random_rotation (bool): If True pointclouds will be rotated randomly.

        """

        # Preconditions.
        assert os.path.exists(dataset_path), "dataset_path must exist: " + str(dataset_path)
        assert isinstance(input_type, str), "input_type must be string: " + str(input_type)
        assert isinstance(output_targets, list), "output_targets must be list: " + \
            str(output_targets)
        if input_type == "image":
            assert len(image_target_shape) == 2, "image_target_shape must be 2-dimensional: " + \
                str(image_target_shape)
        if input_type == "voxelgrid":
            assert len(voxelgrid_target_shape) == 3, "voxelgrid_target_shape must be 3-dimensional: " + \
                str(voxelgrid_target_shape)

        # Assign the instance-variables.
        self.dataset_path = dataset_path
        self.input_type = input_type
        self.output_targets = output_targets
        self.sequence_length = sequence_length
        self.image_target_shape = image_target_shape
        self.voxelgrid_target_shape = voxelgrid_target_shape
        self.voxel_size_meters = voxel_size_meters
        self.voxelgrid_random_rotation = voxelgrid_random_rotation
        self.pointcloud_target_size = pointcloud_target_size
        self.pointcloud_random_rotation = pointcloud_random_rotation

        # Create some caches.
        self.image_cache = {}
        self.voxelgrid_cache = {}
        self.pointcloud_cache = {}

        # Get all the paths.
        self._get_paths()

        # Check if paths are fine.
        if self.input_type == "image":
            assert self.jpg_paths != []
        elif self.input_type == "voxelgrid" or self.input_type == "pointcloud":
            assert self.pcd_paths != []
        else:
            raise Exception("Unexpected: " + self.input_type)
        assert self.json_paths_personal != []
        assert self.json_paths_measures != []

        # Find all QR-codes.
        self._find_qrcodes()
        assert self.qrcodes != [], "No QR-codes found!"

        # Create the QR-codes dictionary.
        self._create_qrcodes_dictionary()

    def _get_paths(self):
        """
        Retrieves all the relevant paths.

        That is: Paths of JPGs, PCDs, and JSONs.
        """

        logging.info(self.dataset_path)

        # Getting the paths for images.
        glob_search_path = os.path.join(self.dataset_path, "storage/person", "**/*.jpg")
        self.jpg_paths = glob2.glob(glob_search_path)

        # Getting the paths for point clouds.
        glob_search_path = os.path.join(self.dataset_path, "storage/person", "**/*.pcd")
        self.pcd_paths = glob2.glob(glob_search_path)

        # Getting the paths for personal and measurement.
        glob_search_path = os.path.join(self.dataset_path, "**/*.json")
        json_paths = glob2.glob(glob_search_path)
        self.json_paths_personal = [
            json_path for json_path in json_paths if "measures" not in json_path]
        self.json_paths_measures = [
            json_path for json_path in json_paths if "measures" in json_path]
        del json_paths

    def _find_qrcodes(self):
        """
        Finds all QR-codes.

        Each individual is represented via a unique QR-codes. This method extracts the set of QR-codes.
        """

        # Go through all the measures and extract their QR-codes.
        qrcodes = []
        for json_path_measure in self.json_paths_measures:
            json_path_measure_file = open(json_path_measure)
            json_data_measure = json.load(json_path_measure_file)
            qrcode = self._extract_qrcode(json_data_measure)
            qrcodes.append(qrcode)
            json_path_measure_file.close()

        # Provide a sorted set.
        qrcodes = sorted(list(set(qrcodes)))

        self.qrcodes = qrcodes

    def _create_qrcodes_dictionary(self):
        """
        Creates a QR-Code-dictionary.

        This basically sorts all PCDs and JPGs.
        With respect to the targets and the QR-Codes.
        This is used heavily during data generation.
        Takes into account timestamps in order to connect data and measures.
        """

        qrcodes_dictionary = {}

        # Go thorugh all measures.
        for json_path_measure in self.json_paths_measures:

            # Load the data and get type.
            measure_file = open(json_path_measure)
            json_data_measure = json.load(measure_file)
            measure_type = json_data_measure["type"]["value"]
            measure_file.close()

            # Ensure manual data. If it is not a manual measurement, skip.
            if measure_type != "manual":
                continue

            # Extract the QR-code.
            qrcode = self._extract_qrcode(json_data_measure)

            # Create an array in the dictionary if necessary.
            if qrcode not in qrcodes_dictionary.keys():
                qrcodes_dictionary[qrcode] = []

            # Extract the targets from the JSON-data.
            targets = self._extract_targets(json_data_measure)

            # Extract the timestamp from the JSON-data.
            timestamp = self._extract_timestamp_from_path(json_path_measure)

            # Filter paths for qrcodes and measurements. Find all JPGs and PCDs for a
            # given QR-code and make sure that the timestamps are related.
            jpg_paths = [jpg_path for jpg_path in self.jpg_paths if self._is_matching_measurement(
                jpg_path, qrcode, timestamp) is True]
            pcd_paths = [pcd_path for pcd_path in self.pcd_paths if self._is_matching_measurement(
                pcd_path, qrcode, timestamp) is True]

            # Store it all.
            qrcodes_dictionary[qrcode].append((targets, jpg_paths, pcd_paths))

        self.qrcodes_dictionary = qrcodes_dictionary

    def _extract_targets(self, json_data_measure):
        """
        Extracts a list of targets from JSON.
        """

        targets = []
        for output_target in self.output_targets:
            value = json_data_measure[output_target]["value"]
            targets.append(value)
        return targets

    def _extract_qrcode(self, json_data_measure):
        """
        Extracts a QR-code from a JSON.
        """

        person_id = json_data_measure["personId"]["value"]
        json_path_personal = [
            json_path for json_path in self.json_paths_personal if person_id in json_path]
        json_path_personal = [
            json_path for json_path in json_path_personal if "ipynb_checkpoints" not in json_path]
        assert len(json_path_personal) == 1, "Found {} jsons for person_id {}\n{}".format(
            len(json_path_personal), person_id, json_path_personal)
        json_path_personal = json_path_personal[0]
        json_data_personal_file = open(json_path_personal)
        json_data_personal = json.load(json_data_personal_file)
        json_data_personal_file.close()
        qrcode = json_data_personal["qrcode"]["value"]
        return qrcode

    def _is_matching_measurement(self, path, qrcode, timestamp, threshold=(60 * 60 * 24 * 1000)):
        """
        Returns True if timetamps match.

        Given a timestamp it extracts a second one from the path.
        It then computes the difference between those two timestamps.
        If the differences are lower than the threshold it is a match.
        And the QR-code must match too.

        Args:
            path (string): Path to some file.
            qrcode (string): A QR-code that is supposed to be related to the file.
            timestamp (string): A timestamp that is supposed to be related to the file.
            threshold (string): A threshold for a match. In milliseconds. Default is one day.

        Returns:
            type: True if it is a match. False otherwise.
        """

        if qrcode not in path:
            return False

        if "measurements" not in path:
            return False

        # Extract the timestamp from the path. Compute difference. Decide.
        path_timestamp = self._extract_timestamp_from_path(path, qrcode)
        difference = abs(int(timestamp) - int(path_timestamp))
        if difference > threshold:
            return False

        return True

    def _extract_timestamp_from_path(self, file_path, qrcode=None):
        """
        Extracts a timestamp from a path.
        """
        # This is fix of the "underscores in QR-code"-issue. Poor man's solution ;)
        if qrcode is not None:
            file_path = file_path.replace(qrcode, "REPLACED")

        timestamp = file_path.split(os.sep)[-1].split("_")[2]
        assert len(timestamp) == 13, "Invalid timestamp {} encountered for {}".format(
            timestamp, file_path)
        assert timestamp.isdigit()
        return timestamp

    def _load_image(self, image_path):
        """
        Loads an image from a given path.

        Makes use of a cache. Ensures that the loaded images has a target size.
        """

        image = self.image_cache.get(image_path, [])
        if image == []:
            image = image_preprocessing.load_img(image_path, target_size=self.image_target_shape)
            image = image.rotate(-90, expand=True)  # Rotation is necessary.
            image = np.array(image)
            self.voxelgrid_cache[image_path] = image
        return image

    def _load_pointcloud(self, pcd_path, preprocess=True, augmentation=True):

        pointcloud = self.pointcloud_cache.get(pcd_path, [])
        if pointcloud == []:
            pointcloud = PyntCloud.from_file(pcd_path).points.values
            pointcloud = np.array(pointcloud)

            if self.pointcloud_target_size is not None and preprocess is True:
                pointcloud = pointcloud[:self.pointcloud_target_size]
                if len(pointcloud) < self.pointcloud_target_size:
                    zeros = np.zeros((self.pointcloud_target_size - len(pointcloud), 4))
                    pointcloud = np.concatenate([pointcloud, zeros])

            if self.pointcloud_random_rotation is True and augmentation is True:
                numpy_points = pointcloud[:, 0:3]
                numpy_points = self._rotate_point_cloud(numpy_points)
                pointcloud[:, 0:3] = numpy_points

            #self.pointcloud_cache[pcd_path] = pointcloud
        return pointcloud

    def _load_voxelgrid(self, pcd_path, preprocess=True, augmentation=True):
        voxelgrid = self.voxelgrid_cache.get(pcd_path, [])
        if voxelgrid == []:

            # Load the pointcloud.
            point_cloud = PyntCloud.from_file(pcd_path)
            if self.voxelgrid_random_rotation is True and augmentation is True:
                points = point_cloud.points
                numpy_points = points.values[:, 0:3]
                numpy_points = self._rotate_point_cloud(numpy_points)
                points.iloc[:, 0:3] = numpy_points
                point_cloud.points = points

            # Create voxelgrid from pointcloud.
            voxelgrid_id = point_cloud.add_structure(
                "voxelgrid",
                size_x=self.voxel_size_meters,
                size_y=self.voxel_size_meters,
                size_z=self.voxel_size_meters)
            voxelgrid = point_cloud.structures[voxelgrid_id].get_feature_vector(mode="density")

            # Do the preprocessing.
            if preprocess is True:
                voxelgrid = utils.ensure_voxelgrid_shape(voxelgrid, self.voxelgrid_target_shape)
                assert voxelgrid.shape == self.voxelgrid_target_shape

            #self.voxelgrid_cache[pcd_path] = voxelgrid # TODO cache is turned off because of you know why...

        return voxelgrid

    def _rotate_point_cloud(self, point_cloud):

        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])

        rotated_data = np.zeros(point_cloud.shape, dtype=np.float32)
        for k in range(point_cloud.shape[0]):

            shape_pc = point_cloud[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

        return rotated_data

    def print_statistics(self):

        for qr_code, array in self.qrcodes_dictionary.items():
            logging.info("QR-Code %s has %d different manual measurements", qr_code, len(array))
            for (targets, jpg_paths, pcd_paths) in array:
                logging.info(" Target %s with %d JPGs and %d PCDs.", targets, len(jpg_paths), len(pcd_paths))

        #qrcodes_dictionary[qrcode].append((targets, jpg_paths, pcd_paths))

    def get_input_shape(self):

        if self.input_type == "image":
            return (90, 160, 3)

        elif self.input_type == "voxelgrid":
            return (32, 32, 32)

        elif self.input_type == "pointcloud":
            return (32000, 4)

        else:
            raise Exception("Unknown input_type: " + self.input_type)

    def get_output_size(self):

        return len(self.output_targets)

    def generate(self, size, qrcodes_to_use=None, verbose=False, yield_file_paths=False, multiprocessing_jobs=1):

        if qrcodes_to_use is None:
            qrcodes_to_use = self.qrcodes

        # Main loop.
        while True:

            # Use only a single process.
            if multiprocessing_jobs == 1:
                yield generate_data(self, size, qrcodes_to_use, verbose, yield_file_paths, None)

            # Use multiple processes.
            elif multiprocessing_jobs > 1:

                # Create chunks of almost equal size.
                subset_sizes = [0] * multiprocessing_jobs
                subset_sizes[0:multiprocessing_jobs - 1] = [size
                                                            // multiprocessing_jobs] * (multiprocessing_jobs - 1)
                subset_sizes[multiprocessing_jobs - 1] = size - \
                    sum(subset_sizes[0:multiprocessing_jobs - 1])
                subset_sizes = [s for s in subset_sizes if s > 0]
                assert sum(subset_sizes) == size

                # Create an output_queue.
                output_queue = mp.Queue()

                # Create the processes.
                processes = []
                for subset_size in subset_sizes:
                    process_target = generate_data
                    process_args = (self, subset_size, qrcodes_to_use,
                                    verbose, yield_file_paths, output_queue)
                    process = mp.Process(target=process_target, args=process_args)
                    processes.append(process)

                # Start the processes.
                for p in processes:
                    p.start()

                # Exit the completed processes.
                for p in processes:
                    p.join()

                # Get process results from the output queue
                output_paths = [output_queue.get() for p in processes]

                # Merge data.
                x_inputs_arrays = []
                y_outputs_arrays = []
                file_paths_arrays = []
                for output_path in output_paths:
                    # Read data from file and delete it.
                    result_values = pickle.load(open(output_path, "rb"))
                    assert len(result_values[0]) > 0
                    assert len(result_values[1]) > 0
                    os.remove(output_path)

                    # Gather the data into arrays.
                    if x_inputs_arrays != []:
                        assert result_values[0].shape[1:] == x_inputs_arrays[-1].shape[1:], str(
                            result_values[0].shape) + " vs " + str(x_inputs_arrays[-1].shape)
                    if y_outputs_arrays != []:
                        assert result_values[1].shape[1:] == y_outputs_arrays[-1].shape[1:], str(
                            result_values[1].shape) + " vs " + str(y_outputs_arrays[-1].shape)
                    x_inputs_arrays.append(result_values[0])
                    y_outputs_arrays.append(result_values[1])
                    if yield_file_paths is True:
                        file_paths_arrays.append(result_values[2])
                    else:
                        file_paths_arrays.append([])

                x_inputs = np.concatenate(x_inputs_arrays)
                y_outputs = np.concatenate(y_outputs_arrays)
                file_paths = np.concatenate(file_paths_arrays)
                assert len(x_inputs) == size
                assert len(y_outputs) == size

                # Done.
                if yield_file_paths is False:
                    yield x_inputs, y_outputs
                else:
                    yield x_inputs, y_outputs, file_paths

            else:
                raise Exception("Unexpected value for 'multiprocessing_jobs' "
                                + str(multiprocessing_jobs))

    def generate_dataset(self, qrcodes_to_use=None):

        if qrcodes_to_use is None:
            qrcodes_to_use = self.qrcodes

        x_qrcodes = []
        x_inputs = []
        y_outputs = []
        for index, qrcode in enumerate(qrcodes_to_use):

            logging.info("Processing: %s", qrcode)

            # Get targets and paths.
            if qrcode not in self.qrcodes_dictionary.keys():
                logging.info("No data for: %s", qrcode)
                continue
            targets, jpg_paths, pcd_paths = self.qrcodes_dictionary[qrcode]
            logging.info(targets)

            # Process image.
            if self.input_type == "image":

                for jpg_path in jpg_paths:
                    image = self._load_image(jpg_path)
                    x_qrcodes.append(qrcode)
                    x_inputs.append(image)
                    y_outputs.append(targets)

            # Process voxelgrid.
            elif self.input_type == "voxelgrid":

                for pcd_path in pcd_paths:
                    try:
                        voxelgrid = self._load_voxelgrid(pcd_path)
                    except Exception as e:
                        print(e)
                        print("Error:", pcd_path)

                    x_qrcodes.append(qrcode)
                    x_inputs.append(voxelgrid)
                    y_outputs.append(targets)

            # Process pointcloud.
            elif self.input_type == "pointcloud":

                for pcd_path in pcd_paths:
                    try:
                        pointcloud = self._load_pointcloud(pcd_path)
                    except Exception as e:
                        print(e)
                        print("Error:", pcd_path)
                        continue

                    x_qrcodes.append(qrcode)
                    x_inputs.append(pointcloud)
                    y_outputs.append(targets)

            else:
                raise Exception("Unknown input_type: " + self.input_type)

        x_qrcodes = np.array(x_qrcodes)
        x_inputs = np.array(x_inputs)
        y_outputs = np.array(y_outputs)

        return x_qrcodes, x_inputs, y_outputs

    def analyze_files(self):

        logging.info("Number of JPGs: %d", len(self.jpg_paths))
        logging.info("Number of PCDs: %d", len(self.pcd_paths))
        logging.info("Number of JSONs (personal): %d", len(self.json_paths_personal))
        logging.info("Number of JSONs (measures): %d", len(self.json_paths_measures))

    def analyze_targets(self):
        """
        Extracts and analyzes all targets from the dataset.
        """

        all_targets = []
        for _, targets_array in self.qrcodes_dictionary.items():
            for (targets, _, _) in targets_array:
                all_targets.append(targets)

        x = [targets[0] for targets in all_targets]
        y = [targets[1] for targets in all_targets]
        plt.scatter(x, y)
        plt.xlabel(self.output_targets[0])
        plt.ylabel(self.output_targets[1])
        plt.title("Distribution of " + str(len(all_targets)) + " targets.")
        plt.show()
        plt.close()

    def analyze_pointclouds(self):

        logging.info("Analyzing pointclouds...")
        pointcloud_sizes = []
        bar = progressbar.ProgressBar(max_value=len(self.pcd_paths))
        for index, pcd_path in enumerate(self.pcd_paths):
            try:
                pointcloud = self._load_pointcloud(pcd_path, preprocess=False, augmentation=False)
                pointcloud_sizes.append(pointcloud.shape[0])
            except ValueError:
                pass
            bar.update(index)
        bar.finish()

        logging.info("Rendering histogram...")
        plt.hist(pointcloud_sizes)
        #plt.xlabel(self.output_targets[0])
        #plt.ylabel(self.output_targets[1])
        plt.title("Distribution of pointclouds-sizes.")
        plt.show()
        plt.close()

    def analyze_voxelgrids(self):

        logging.info("Analyzing voxelgrids...")
        voxelgrid_sizes = [0] * len(self.pcd_paths)
        numbers_of_voxels = [0] * len(self.pcd_paths)
        voxel_densities = [0] * len(self.pcd_paths)
        bar = progressbar.ProgressBar(max_value=len(self.pcd_paths))
        for index, pcd_path in enumerate(self.pcd_paths):
            try:
                voxelgrid = self._load_voxelgrid(pcd_path, augmentation=False, preprocess=False)
                voxelgrid_size = voxelgrid.shape[0]
                voxelgrid_sizes[index] = voxelgrid_size
                number_of_voxels = np.count_nonzero(voxelgrid != 0.0)
                numbers_of_voxels[index] = number_of_voxels
                voxel_densities[index] = number_of_voxels / (voxelgrid_size ** 3)
            except ValueError as error:
                print(pcd_path)
                print(error)
            bar.update(index)
        bar.finish()

        logging.info("Getting the PCDs with the lowest voxel densities...")
        argsort = np.argsort(numbers_of_voxels)
        for index in argsort[0:20]:
            logging.info("%.2f : %.2f", numbers_of_voxels[index], self.pcd_paths[index])

        logging.info("Rendering histograms...")
        plt.hist(voxelgrid_sizes)
        plt.title("Distribution of voxelgrid-sizes.")
        plt.show()
        plt.close()

        plt.hist(numbers_of_voxels)
        plt.title("Distribution of number of voxels.")
        plt.show()
        plt.close()

        plt.hist(voxel_densities)
        plt.title("Distribution of voxel densities.")
        plt.show()
        plt.close()


def test_generator():

    if os.path.exists("datasetpath.txt"):
        dataset_path = open("datasetpath.txt", "r").read().replace("\n", "")
    else:
        dataset_path = "../data"

    data_generator = DataGenerator(dataset_path=dataset_path,
                                   input_type="pointcloud", output_targets=["height", "weight"])

    logging.info("jpg_paths %d", len(data_generator.jpg_paths))
    logging.info("pcd_paths %d", len(data_generator.jpg_paths))
    logging.info("json_paths_personal %d", len(data_generator.jpg_paths))
    logging.info("json_paths_measures %d", len(data_generator.jpg_paths))
    logging.info("QR-Codes: \n" + "\n".join(data_generator.qrcodes))
    #logging.info(data_generator.qrcodes_dictionary)
    data_generator.print_statistics()

    qrcodes_shuffle = list(data_generator.qrcodes)
    random.shuffle(qrcodes_shuffle)
    split_index = int(0.8 * len(qrcodes_shuffle))
    qrcodes_train = qrcodes_shuffle[:split_index]
    qrcodes_validate = qrcodes_shuffle[split_index:]

    logging.info("Training data:")
    x_train, y_train = next(data_generator.generate(
        size=200, qrcodes_to_use=qrcodes_train, verbose=True))
    logging.info(x_train.shape)
    logging.info(y_train.shape)
    logging.info("")

    logging.info("Validation data:")
    x_validate, y_validate = next(data_generator.generate(
        size=20, qrcodes_to_use=qrcodes_validate, verbose=True))
    logging.info(x_validate.shape)
    logging.info(y_validate.shape)
    logging.info("")


def test_dataset():

    if os.path.exists("datasetpath.txt"):
        dataset_path = open("datasetpath.txt", "r").read().replace("\n", "")
    else:
        dataset_path = "../data"

    data_generator = DataGenerator(dataset_path=dataset_path,
                                   input_type="image", output_targets=["height", "weight"])

    x_qrcodes, x_inputs, y_outputs = data_generator.generate_dataset(data_generator.qrcodes[0:4])
    logging.info(len(x_qrcodes))


def test_parameters():

    if os.path.exists("datasetpath.txt"):
        dataset_path = open("datasetpath.txt", "r").read().replace("\n", "")
    else:
        dataset_path = "../data"

    logging.info("Testing image...")
    data_generator = DataGenerator(dataset_path=dataset_path, input_type="image", output_targets=[
                                   "height", "weight"], image_target_shape=(20, 20))
    x_input, y_output = next(data_generator.generate(size=1))
    assert x_input.shape[1:3] == (20, 20)

    logging.info("Testing voxelgrid...")
    data_generator = DataGenerator(dataset_path=dataset_path, input_type="voxelgrid", output_targets=[
                                   "height", "weight"], voxelgrid_target_shape=(20, 20, 20))
    x_input, y_output = next(data_generator.generate(size=1))
    assert x_input.shape[1:] == (20, 20, 20)

    logging.info("Testing pointcloud...")
    data_generator = DataGenerator(dataset_path=dataset_path, input_type="pointcloud", output_targets=[
                                   "height", "weight"], pointcloud_target_size=16000)
    x_input, y_output = next(data_generator.generate(size=1))
    assert x_input.shape[1:] == (20, 4)

    logging.info("Done.")


def generate_data(class_self, size, qrcodes_to_use, verbose, yield_file_paths, output_queue):
    if verbose is True:
        logging.info("Generating QR-codes to be used: %s", qrcodes_to_use)

    assert size != 0

    x_inputs = []
    y_outputs = []
    file_paths = []

    if verbose is True:
        bar = progressbar.ProgressBar(max_value=size)
    while len(x_inputs) < size:

        # Get a random QR-code.
        qrcode = random.choice(qrcodes_to_use)

        # Get targets and paths randomly.
        if qrcode not in class_self.qrcodes_dictionary.keys():
            continue
        targets, jpg_paths, pcd_paths = random.choice(class_self.qrcodes_dictionary[qrcode])

        # No pointclouds.
        if len(pcd_paths) == 0:
            continue

        # Get a sample.
        x_input = None
        y_output = None
        file_path = None

        # Get the input. Not dealing with sequences.
        if class_self.sequence_length == 0:
            x_input, file_path = get_input(class_self, jpg_paths, pcd_paths)

        # Get the input. Dealing with sequences here.
        else:
            count = 0
            x_input, file_path = [], []
            while count != class_self.sequence_length:
                x, f = get_input(class_self, jpg_paths, pcd_paths)
                if x is not None and f is not None:
                    x_input.append(x)
                    file_path.append(f)
                    count += 1
            x_input = np.array(x_input)
            file_path = np.array(file_path)

        # Set the output.
        y_output = targets

        # Got a proper sample.
        if x_input is not None and y_output is not None and file_path is not None:
            x_inputs.append(x_input)
            y_outputs.append(y_output)
            file_paths.append(file_path)

        assert len(x_inputs) == len(y_outputs)
        assert len(y_outputs) == len(file_paths)

        if verbose is True:
            bar.update(len(x_inputs))

    if verbose is True:
        bar.finish()

    assert len(x_inputs) == size
    assert len(y_outputs) == size

    # Turn everything into ndarrays.
    x_inputs = np.array(x_inputs)
    y_outputs = np.array(y_outputs)

    # Prepare result values.
    assert len(x_inputs) == size
    assert len(y_outputs) == size
    if yield_file_paths is False:
        return_values = (x_inputs, y_outputs)
    else:
        return_values = (x_inputs, y_outputs, file_paths)

    # This is used in multiprocessing. Creates a pickle file and puts the data there.
    if output_queue is not None:
        output_path = uuid.uuid4().hex + ".p"
        pickle.dump(return_values, open(output_path, "wb"))
        output_queue.put(output_path)
    else:
        return return_values


def get_input(class_self, jpg_paths, pcd_paths):
    # Get a random image.
    if class_self.input_type == "image":
        if len(jpg_paths) == 0:
            logging.info("777")
            return None, None
        jpg_path = random.choice(jpg_paths)
        image = class_self._load_image(jpg_path)
        file_path = jpg_path
        x_input = image

    # Get a random voxelgrid.
    elif class_self.input_type == "voxelgrid":
        if len(pcd_paths) == 0:
            return None, None
        pcd_path = random.choice(pcd_paths)
        try:
            voxelgrid = class_self._load_voxelgrid(pcd_path)
            file_path = pcd_path
            x_input = voxelgrid
        except Exception as e:
            print(e)
            return None, None

    # Get a random pointcloud.
    elif class_self.input_type == "pointcloud":
        if len(pcd_paths) == 0:
            return None, None
        pcd_path = random.choice(pcd_paths)
        try:
            pointcloud = class_self._load_pointcloud(pcd_path)
            file_path = pcd_path
            x_input = pointcloud
        except Exception as e:
            print(e)
            return None, None

    # Should not happen.
    else:
        raise Exception("Unknown input_type: " + class_self.input_type)

    return x_input, file_path


def create_datagenerator_from_parameters(dataset_path, dataset_parameters):
    logging.info("Creating data-generator...")
    datagenerator = DataGenerator(
        dataset_path=dataset_path,
        input_type=dataset_parameters["input_type"],
        output_targets=dataset_parameters["output_targets"],
        sequence_length=dataset_parameters.get("sequence_length", 0),
        voxelgrid_target_shape=dataset_parameters.get("voxelgrid_target_shape", None),
        voxel_size_meters=dataset_parameters.get("voxel_size_meters", None),
        voxelgrid_random_rotation=dataset_parameters.get("voxelgrid_random_rotation", None),
        pointcloud_target_size=dataset_parameters.get("pointcloud_target_size", None),
        pointcloud_random_rotation=dataset_parameters.get("pointcloud_random_rotation", None)
    )
    #datagenerator.print_statistics()
    return datagenerator


def get_dataset_path():
    if os.path.exists("datasetpath.txt"):
        with open("datasetpath.txt", "r") as file:
            dataset_path = file.read().replace("\n", "")
    else:
        dataset_path = "../data"
    return dataset_path


if __name__ == "__main__":
    test_generator()
    #test_dataset()
    #test_parameters()
