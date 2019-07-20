'''
This file contains a data-generator that the preprocessed data. Note: It does not work on ETL-data.

Note: It currently works on pointcloud-data only!
'''


from __future__ import absolute_import
import os
import numpy as np
import glob2 as glob
import random
import progressbar
from pyntcloud import PyntCloud
import shutil
#import matplotlib.pyplot as plt
#import multiprocessing as mp
#import uuid
import pickle
from . import utils


class PreprocessedDataGenerator(object):
    """
    This class generates data for training.
    """

    def __init__(
        self,
        dataset_path,
        input_type,
        output_targets=["height"],
        filter=None,
        sequence_length=0,
        image_target_shape=(160, 90),
        voxelgrid_target_shape=(32, 32, 32),
        voxel_size_meters=0.01,
        voxelgrid_random_rotation=False,
        pointcloud_target_size=32000,
        pointcloud_random_rotation=False,
        rgbmap_target_width=512, 
        rgbmap_target_height=512, 
        rgbmap_scale_factor=1.5,
        rgbmap_axis = "vertical"
        ):
        """
        Initializes a DataGenerator.

        Args:
            dataset_path (string): Where the raw data is.
            input_type (string): Specifies how the input-data for the Neural Network looks like. Either 'image', 'pointcloud', 'voxgrid'.
            output_targets (list of strings): A list of targets for the Neural Network. For example *['height', 'weight']*.
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
            assert len(image_target_shape) == 2, "image_target_shape must be 2-dimensional: " + str(image_target_shape)
        if input_type == "voxelgrid":
            assert len(voxelgrid_target_shape) == 3, "voxelgrid_target_shape must be 3-dimensional: " + str(voxelgrid_target_shape)


        # Set the dataset-path.
        if input_type == "image":
            self.dataset_path = os.path.join(dataset_path, "jpg")
        elif input_type == "voxelgrid":
            self.dataset_path = os.path.join(dataset_path, "pcd")
        elif input_type == "pointcloud":
            self.dataset_path = os.path.join(dataset_path, "pcd")
        elif input_type == "rgbmap":
            self.dataset_path = os.path.join(dataset_path, "pcd")
        else:
            raise Exception("Unknown input_type: " + class_self.input_type)
        
        # Assign the instance-variables.
        self.input_type = input_type
        self.output_targets = output_targets
        self.filter = filter
        self.sequence_length = sequence_length
        self.image_target_shape = image_target_shape
        self.voxelgrid_target_shape = voxelgrid_target_shape
        self.voxel_size_meters = voxel_size_meters
        self.voxelgrid_random_rotation = voxelgrid_random_rotation
        self.pointcloud_target_size = pointcloud_target_size
        self.pointcloud_random_rotation = pointcloud_random_rotation
        self.rgbmap_target_width = rgbmap_target_width
        self.rgbmap_target_height = rgbmap_target_height
        self.rgbmap_scale_factor = rgbmap_scale_factor
        self.rgbmap_axis = rgbmap_axis
        
        # Find all QR-codes.
        self._find_qrcodes()
        assert self.qrcodes != [], "No QR-codes found!"

        # Prepare the data.
        self._prepare_qrcodes_dictionary()

        # Create some caches.
        #self.image_cache = {}
        #self.voxelgrid_cache = {}
        #self.pointcloud_cache = {}
        
        # Check if paths are fine.
        #if self.input_type == "image":
        #    assert self.all_jpg_paths != []
        #elif self.input_type == "voxelgrid" or self.input_type == "pointcloud":
        #    assert self.all_pcd_paths != []
        #else:
        #    raise Exception("Unexpected: " + self.input_type)


    def _find_qrcodes(self):
        """
        Finds all QR-codes.

        Each individual is represented via a unique QR-code. This method extracts the set of QR-codes.
        """

        # Retrieve the QR-codes from the folders.
        paths = glob.glob(os.path.join(self.dataset_path, "*"))
        paths = [path for path in paths if os.path.isdir(path) == True]
        self.qrcodes = sorted([path.split("/")[-1] for path in paths])

        
    def _prepare_qrcodes_dictionary(self):

        self.all_pcd_paths = []
        self.all_jpg_paths = []
        
        self.qrcodes_dictionary = {}
        for qrcode in self.qrcodes:
            # Getting all files that belong to the QR-code.
            glob_search_path = os.path.join(self.dataset_path, qrcode)
            preprocessed_paths = glob.glob(os.path.join(glob_search_path, "*.p"))
            
            assert len(preprocessed_paths) != 0, "ERROR: No files found at {}!".format(glob_search_path)
            
            # Filter the paths if specified.
            if self.filter != None:
                if self.filter == "front":
                    filter_for = "104"
                elif self.filter == "360":
                    filter_for = "107"
                elif self.filter == "back":
                    filter_for = "110"
                preprocessed_paths = [path for path in preprocessed_paths if os.path.basename(path).split("_")[-2] == filter_for]

            # Done.
            self.qrcodes_dictionary[qrcode] = preprocessed_paths
            


    def analyze_files(self):

        for qrcode in self.qrcodes:
            print("QR-code:", qrcode)
            print("  Number of samples: {}".format(len(self.qrcodes_dictionary[qrcode])))

        
    def generate(self, size, qrcodes_to_use=None, verbose=False, multiprocessing_jobs=1):

        if qrcodes_to_use == None:
            qrcodes_to_use = self.qrcodes

        # Main loop.
        while True:

            # Use only a single process.
            if multiprocessing_jobs == 1:
                yield generate_data(self, size, qrcodes_to_use, verbose, None)

            # Use multiple processes.
            elif multiprocessing_jobs > 1:

                # Create chunks of almost equal size.
                subset_sizes = [0] * multiprocessing_jobs
                subset_sizes[0:multiprocessing_jobs - 1] = [size // multiprocessing_jobs] * (multiprocessing_jobs - 1)
                subset_sizes[multiprocessing_jobs - 1] = size - sum(subset_sizes[0:multiprocessing_jobs - 1])
                subset_sizes = [s for s in subset_sizes if s > 0]
                assert sum(subset_sizes) == size

                # Create an output_queue.
                output_queue = mp.Queue()

                # Create the processes.
                processes = []
                for subset_size in subset_sizes:
                    process_target = generate_data
                    process_args = (self, subset_size, qrcodes_to_use, verbose, yield_file_paths, output_queue)
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
                        assert result_values[0].shape[1:] == x_inputs_arrays[-1].shape[1:], str(result_values[0].shape) + " vs " + str(x_inputs_arrays[-1].shape)
                    if y_outputs_arrays != []:
                        assert result_values[1].shape[1:] == y_outputs_arrays[-1].shape[1:], str(result_values[1].shape) + " vs " + str(y_outputs_arrays[-1].shape)
                    x_inputs_arrays.append(result_values[0])
                    y_outputs_arrays.append(result_values[1])
                    if yield_file_paths == True:
                        file_paths_arrays.append(result_values[2])
                    else:
                        file_paths_arrays.append([])

                x_inputs = np.concatenate(x_inputs_arrays)
                y_outputs = np.concatenate(y_outputs_arrays)
                file_paths = np.concatenate(file_paths_arrays)
                assert len(x_inputs) == size
                assert len(y_outputs) == size

                # Done.
                if yield_file_paths == False:
                    yield x_inputs, y_outputs
                else:
                    yield x_inputs, y_outputs, file_paths

            else:
                raise Exception("Unexpected value for 'multiprocessing_jobs' " + str(multiprocessing_jobs))
                

    #def _load_pointcloud(self, pcd_path, preprocess=True, augmentation=True):

    #    pointcloud = self.pointcloud_cache.get(pcd_path, [])
    #    if pointcloud == []:
    #        pointcloud = PyntCloud.from_file(pcd_path).points.values

    #        if self.pointcloud_target_size != None and preprocess == True:
    #            pointcloud = np.array(pointcloud)[:,0:3] # Drop confidence.
    #            pointcloud = pointcloud[:self.pointcloud_target_size]
    #            if len(pointcloud) < self.pointcloud_target_size:
    #                zeros = np.zeros((self.pointcloud_target_size - len(pointcloud), 4))
    #                pointcloud = np.concatenate([pointcloud, zeros])

    #        if self.pointcloud_random_rotation == True and augmentation==True:
    #            numpy_points = pointcloud[:,0:3]
    #            numpy_points = self._rotate_point_cloud(numpy_points)
    #            pointcloud[:,0:3] = numpy_points

            #self.pointcloud_cache[pcd_path] = pointcloud
    #    return pointcloud

    def _create_voxelgrid_from_pointcloud(self, pointcloud, augmentation=True):
        if self.voxelgrid_random_rotation == True and augmentation == True:
            pointcloud = self._rotate_point_cloud(pointcloud)

        # Create voxelgrid from pointcloud.
        pointcloud = PyntCloud(pointcloud)
        voxelgrid_id = pointcloud.add_structure("voxelgrid", size_x=self.voxel_size_meters, size_y=self.voxel_size_meters, size_z=self.voxel_size_meters)
        voxelgrid = pointcloud.structures[voxelgrid_id].get_feature_vector(mode="density")

        # Do the preprocessing.
        if preprocess == True:
            voxelgrid = utils.ensure_voxelgrid_shape(voxelgrid, self.voxelgrid_target_shape)
            assert voxelgrid.shape == self.voxelgrid_target_shape

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
    print("Creating data-generator...")
    datagenerator = PreprocessedDataGenerator(
        dataset_path=dataset_path,
        input_type=dataset_parameters["input_type"],
        output_targets=dataset_parameters["output_targets"],
        filter=dataset_parameters.get("filter", None),
        sequence_length=dataset_parameters.get("sequence_length", 0),
        voxelgrid_target_shape=dataset_parameters.get("voxelgrid_target_shape", None),
        voxel_size_meters=dataset_parameters.get("voxel_size_meters", None),
        voxelgrid_random_rotation=dataset_parameters.get("voxelgrid_random_rotation", None),
        pointcloud_target_size=dataset_parameters.get("pointcloud_target_size", None),
        pointcloud_random_rotation=dataset_parameters.get("pointcloud_random_rotation", None),
        rgbmap_target_width=dataset_parameters.get("rgbmap_target_width", None),
        rgbmap_target_height=dataset_parameters.get("rgbmap_target_height", None),
        rgbmap_scale_factor=dataset_parameters.get("rgbmap_scale_factor", None),
        rgbmap_axis=dataset_parameters.get("rgbmap_axis", None),
    )
    #datagenerator.print_statistics()
    return datagenerator


def get_dataset_path(root_path="/localssd/preprocessed"):
    if os.path.exists("etldatasetpath.txt"):
        with open("etldatasetpath.txt", "r") as file:
            dataset_path = file.read().replace("\n", "")
    else:
        # Finding the latest.
        dataset_paths = glob.glob(os.path.join(root_path, "*"))
        dataset_paths = [dataset_path for dataset_path in dataset_paths if os.path.isdir(dataset_path)]
        dataset_path = list(reversed(sorted(dataset_paths)))[0]

    return dataset_path


def generate_data(class_self, size, qrcodes_to_use, verbose, output_queue):
    if verbose == True:
        print("Generating using QR-codes:", qrcodes_to_use)

    assert size != 0

    x_inputs = []
    y_outputs = []

    if verbose == True:
        bar = progressbar.ProgressBar(max_value=size)
    while len(x_inputs) < size:

        # Get a random QR-code.
        qrcode = random.choice(qrcodes_to_use)

        # Get targets and paths randomly.
        if qrcode not in  class_self.qrcodes_dictionary.keys():
            continue
            
        # Get a sample.
        x_input = None
        y_output = None

        # Get the input. Not dealing with sequences.
        if class_self.sequence_length == 0:
            if len(class_self.qrcodes_dictionary[qrcode]) > 0:
                preprocessed_path = random.choice(class_self.qrcodes_dictionary[qrcode])
                with open(preprocessed_path, "rb") as file:
                    (pointcloud, targets) = load_pointcloud_and_target(file, class_self.output_targets)
                    assert pointcloud.shape[0] != 0, "Empty pointcloud in file {}.".format(preprocessed_path)

                x_input = get_input(class_self, pointcloud)
                y_output = targets

        # Get the input. Dealing with sequences here.
        else:
            preprocessed_paths = np.array(class_self.qrcodes_dictionary[qrcode])
            
            # Do not touch the QR-code if it does not have enough samples.
            if len(preprocessed_paths) < class_self.sequence_length:
                continue
            
            # Get some random indices that are in order.
            indices = np.arange(len(preprocessed_paths))
            np.random.shuffle(indices)
            indices = indices[:class_self.sequence_length]
            indices = np.sort(indices)

            preprocessed_paths = preprocessed_paths[indices]
            x_input, file_path = [], []
            for preprocessed_path in preprocessed_paths:
                with open(preprocessed_path, "rb") as file:
                    (pointcloud, targets) = load_pointcloud_and_target(file, class_self.output_targets)
                    assert pointcloud.shape[0] != 0, "Empty pointcloud in file {}.".format(preprocessed_path)
                    try:
                        x_input.append(get_input(class_self, pointcloud))
                    except:
                        print(pointcloud.shape, preprocessed_path)
                        exit(0)
                    file_path.append(preprocessed_path)
            
            x_input = np.array(x_input)
            y_output = targets
 
        # Got a proper sample.
        if x_input is not None and y_output is not None:
            x_inputs.append(x_input)
            y_outputs.append(y_output)

        assert len(x_inputs) == len(y_outputs)

        if verbose == True:
            bar.update(len(x_inputs))

    if verbose == True:
        bar.finish()

    assert len(x_inputs) == size
    assert len(y_outputs) == size

    # Turn everything into ndarrays.
    x_inputs = np.array(x_inputs)
    y_outputs = np.array(y_outputs)

    # Prepare result values.
    assert len(x_inputs) == size
    assert len(y_outputs) == size
    return_values =  (x_inputs, y_outputs)

    # This is used in multiprocessing. Creates a pickle file and puts the data there.
    if output_queue != None:
        output_path = uuid.uuid4().hex + ".p"
        pickle.dump(return_values, open(output_path, "wb"))
        output_queue.put(output_path)
    else:
        return return_values


def load_pointcloud_and_target(file, output_targets):
    (pointcloud, targets) = pickle.load(file)
    if output_targets == ["height"]:
        targets = targets[0:1]
    elif output_targets == ["weight"]:
        targets = targets[1:]
    return (pointcloud, targets)
    

def get_input(class_self, pointcloud):
    
    # Get a random image.
    if class_self.input_type == "image":
        raise Exception("Not expected to work with image-data.")

    # Get a random voxelgrid.
    elif class_self.input_type == "voxelgrid":
        voxelgrid = class_self._create_voxelgrid_from_pointcloud(pointcloud)
        x_input = voxelgrid
 
    # Get a random pointcloud.
    elif class_self.input_type == "pointcloud":
        pointcloud = utils.subsample_pointcloud(pointcloud, class_self.pointcloud_target_size)
        x_input = pointcloud
  
    # Get a random pointcloud.
    elif class_self.input_type == "rgbmap":
        rgb_map = utils.pointcloud_to_rgb_map(
            pointcloud, 
            class_self.rgbmap_target_width, 
            class_self.rgbmap_target_height,
            class_self.rgbmap_scale_factor,
            class_self.rgbmap_axis
        )
        x_input = rgb_map

    # Should not happen.
    else:
        raise Exception("Unknown input_type: " + class_self.input_type)

    return x_input