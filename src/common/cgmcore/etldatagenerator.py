'''
This file contains a data-generator that works on the data as it is provided by the ETL.
'''

from __future__ import absolute_import
import os
import logging
import logging.config
import numpy as np
import glob2 as glob
import random
import uuid
import progressbar
import multiprocessing as mp
from pyntcloud import PyntCloud
import pickle
from . import utils

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


class ETLDataGenerator(object):
    """
    This class generates data for training.
    """

    def __init__(
        self,
        dataset_path,
        input_type,
        #output_targets,
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
        #assert isinstance(output_targets, list), "output_targets must be list: " + str(output_targets)
        if input_type == "image":
            assert len(image_target_shape) == 2, "image_target_shape must be 2-dimensional: " + \
                str(image_target_shape)
        if input_type == "voxelgrid":
            assert len(voxelgrid_target_shape) == 3, "voxelgrid_target_shape must be 3-dimensional: " + \
                str(voxelgrid_target_shape)

        # Assign the instance-variables.
        self.dataset_path = dataset_path
        self.input_type = input_type
        #self.output_targets = output_targets
        self.sequence_length = sequence_length
        self.image_target_shape = image_target_shape
        self.voxelgrid_target_shape = voxelgrid_target_shape
        self.voxel_size_meters = voxel_size_meters
        self.voxelgrid_random_rotation = voxelgrid_random_rotation
        self.pointcloud_target_size = pointcloud_target_size
        self.pointcloud_random_rotation = pointcloud_random_rotation

        # Find all QR-codes.
        self._find_qrcodes()
        assert self.qrcodes is not None, "No QR-codes found!"

        # Prepare the data.
        self._prepare_qrcodes_dictionary()

        # Create some caches.
        self.image_cache = {}
        self.voxelgrid_cache = {}
        self.pointcloud_cache = {}

        # Check if paths are fine.
        if self.input_type == "image":
            assert self.all_jpg_paths != []
        elif self.input_type == "voxelgrid" or self.input_type == "pointcloud":
            assert self.all_pcd_paths != []
        else:
            raise Exception("Unexpected: " + self.input_type)

    def _find_qrcodes(self):
        """
        Finds all QR-codes.

        Each individual is represented via a unique QR-code. This method extracts the set of QR-codes.
        """

        # Retrieve the QR-codes from the folders.
        paths = glob.glob(os.path.join(self.dataset_path, "*"))
        self.qrcodes = sorted([path.split("/")[-1] for path in paths])

    def _prepare_qrcodes_dictionary(self):

        self.all_pcd_paths = []
        self.all_jpg_paths = []

        self.qrcodes_dictionary = {}
        for qrcode in self.qrcodes:
            self.qrcodes_dictionary[qrcode] = []
            #logging.info("QR-code:", qrcode)
            measurement_paths = glob.glob(os.path.join(self.dataset_path, qrcode, "*"))
            for measurement_path in measurement_paths:
                # Getting PCDs.
                pcd_paths = glob.glob(os.path.join(measurement_path, "pcd", "*.pcd"))

                # Getting JPGs.
                jpg_paths = glob.glob(os.path.join(measurement_path, "jpg", "*.jpg"))

                # Loading the targets.
                target_path = os.path.join(measurement_path, "target.txt")
                target_file = open(target_path, "r")
                targets = np.array([float(value) for value in target_file.read().split(",")])
                target_file.close()

                # Done.
                self.qrcodes_dictionary[qrcode].append((pcd_paths, jpg_paths, targets))
                self.all_pcd_paths.extend(pcd_paths)
                self.all_jpg_paths.extend(jpg_paths)

    def analyze_files(self):

        for qrcode in self.qrcodes:
            logging.info("QR-code: %s", qrcode)
            for pcd_paths, jpg_paths, targets in self.qrcodes_dictionary[qrcode]:
                logging.info(" %d PCD-files for targets %s", len(pcd_paths), targets)
                logging.info(" %d JPG-files for targets %s", len(jpg_paths), targets)

        logging.info("Total PCD-files: %d", len(self.all_pcd_paths))
        logging.info("Total JPG-files: %d", len(self.all_jpg_paths))

        #measurement_paths = glob.glob(os.path.join(self.dataset_path, qrcode, "*"))
        #for measurement_path in measurement_paths:
        #    logging.info("  %s", measurement_path.split("/")[-1])
        #    pcd_paths = glob.glob(os.path.join(measurement_path, "*.pcd"))
        #    logging.info("    %d PCDs", len(pcd_paths))
        #logging.info("Number of JPGs: %d", len(self.jpg_paths))
        #logging.info("Number of PCDs: %d", len(self.pcd_paths))
        #logging.info("Number of JSONs (personal): %d", len(self.json_paths_personal))
        #logging.info("Number of JSONs (measures): %d", len(self.json_paths_measures))

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

    def _load_pointcloud(self, pcd_path, preprocess=True, augmentation=True):

        pointcloud = self.pointcloud_cache.get(pcd_path, [])
        if pointcloud is None:
            pointcloud = PyntCloud.from_file(pcd_path).points.values

            if self.pointcloud_target_size is not None and preprocess is True:
                pointcloud = np.array(pointcloud)[:, 0:3]  # Drop confidence.
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


def create_datagenerator_from_parameters(dataset_path, dataset_parameters):
    logging.info("Creating data-generator...")
    datagenerator = ETLDataGenerator(
        dataset_path=dataset_path,
        input_type=dataset_parameters["input_type"],
        #output_targets=dataset_parameters["output_targets"],
        sequence_length=dataset_parameters.get("sequence_length", 0),
        voxelgrid_target_shape=dataset_parameters.get("voxelgrid_target_shape", None),
        voxel_size_meters=dataset_parameters.get("voxel_size_meters", None),
        voxelgrid_random_rotation=dataset_parameters.get("voxelgrid_random_rotation", None),
        pointcloud_target_size=dataset_parameters.get("pointcloud_target_size", None),
        pointcloud_random_rotation=dataset_parameters.get("pointcloud_random_rotation", None)
    )
    #datagenerator.print_statistics()
    return datagenerator


def get_dataset_path(root_path="../data/etl"):
    if os.path.exists("etldatasetpath.txt"):
        with open("etldatasetpath.txt", "r") as file:
            dataset_path = file.read().replace("\n", "")
    else:
        # Finding the latest.
        dataset_paths = glob.glob(os.path.join(root_path, "*"))
        dataset_paths = [
            dataset_path for dataset_path in dataset_paths if os.path.isdir(dataset_path)]
        dataset_path = list(reversed(sorted(dataset_paths)))[0]

    return dataset_path


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
        pcd_paths, jpg_paths, targets = random.choice(class_self.qrcodes_dictionary[qrcode])

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
