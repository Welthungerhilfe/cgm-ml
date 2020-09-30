import logging
import random
import keras.preprocessing.image as image_preprocessing
from pyntcloud import PyntCloud
import numpy as np
import utils as etl_utils
import cgmcore.utils as core_utils

log = logging.getLogger(__name__)

# TODO solve configuration management

# The output of QR code is to get x-input, y-output and filepath
# output  data from ALL QRcodes must be merged and put together which finally forms the training set


class QRCode:
    def __init__(self, qrcode, input_type, sequence_length,
                 voxelgrid_random_rotation, voxel_size_meters,
                 voxelgrid_target_shape, image_target_shape,
                 pointcloud_target_size, pointcloud_random_rotation):
        """
        given a qr-code, get all relevant data for this code
        serves as a reader for a particular qr code
        # supply a config dict
        :param qrcode:
        """
        self.qrcode = qrcode
        self.input_type = input_type
        self.sequence_length = sequence_length
        self.voxelgrid_random_rotation = voxelgrid_random_rotation
        self.voxel_size_meters = voxel_size_meters
        self.voxelgrid_target_shape = voxelgrid_target_shape
        self.image_target_shape = image_target_shape
        self.pointcloud_target_size = pointcloud_target_size
        self.pointcloud_random_rotation = pointcloud_random_rotation
        self.x_input = None
        self.y_output = None
        self.out_filepath = None

    def get_targets(self):
        pass

    def get_pcd_paths(self):
        pass

    def verify_point_cloud(self):
        pcd_paths = self.get_pcd_paths()
        if len(pcd_paths) == 0:
            log.warn("Ignoring qr code %s as pcd paths is empty" % self.qrcode)
            return

    # TODO clarify cache
    def _load_voxelgrid(self, pcd_path, preprocess=True, augmentation=True):
        # Load the pointcloud.
        point_cloud = PyntCloud.from_file(pcd_path)
        if self.voxelgrid_random_rotation == True and augmentation == True:
            points = point_cloud.points
            numpy_points = points.values[:, 0:3]
            numpy_points = etl_utils._rotate_point_cloud(numpy_points)
            points.iloc[:, 0:3] = numpy_points
            point_cloud.points = points

        # Create voxelgrid from pointcloud.
        voxelgrid_id = point_cloud.add_structure(
            "voxelgrid",
            size_x=self.voxel_size_meters,
            size_y=self.voxel_size_meters,
            size_z=self.voxel_size_meters)
        voxelgrid = point_cloud.structures[voxelgrid_id].get_feature_vector(
            mode="density")
        # Do the preprocessing.
        if preprocess:
            voxelgrid = core_utils.ensure_voxelgrid_shape(
                voxelgrid, self.voxelgrid_target_shape)
            assert voxelgrid.shape == self.voxelgrid_target_shape

        #self.voxelgrid_cache[pcd_path] = voxelgrid # TODO cache is turned off because of you know why...
        return voxelgrid

    def _load_pointcloud(self, pcd_path, preprocess=True, augmentation=True):
        pointcloud = PyntCloud.from_file(pcd_path).points.values
        pointcloud = np.array(pointcloud)

        if self.pointcloud_target_size is not None and preprocess is True:
            pointcloud = pointcloud[:self.pointcloud_target_size]
            if len(pointcloud) < self.pointcloud_target_size:
                zeros = np.zeros(
                    (self.pointcloud_target_size - len(pointcloud), 4))
                pointcloud = np.concatenate([pointcloud, zeros])

        if self.pointcloud_random_rotation and augmentation:
            numpy_points = pointcloud[:, 0:3]
            numpy_points = etl_utils._rotate_point_cloud(numpy_points)
            pointcloud[:, 0:3] = numpy_points

        #self.pointcloud_cache[pcd_path] = pointcloud
        return pointcloud

    def _load_image(self, image_path):
        """
        Loads an image from a given path.

        Makes use of a cache. Ensures that the loaded images has a target size.
        """
        image = image_preprocessing.load_img(
            image_path, target_size=self.image_target_shape)
        image = image.rotate(-90, expand=True)  # Rotation is necessary.
        image = np.array(image)
        return image

    def process(self):
        # process the data and create x-input, y-output, filepaths
        ## TODO must know what are its jpg_paths, pcd_paths
        # TODO must know its targets
        # Set the output.
        y_output = targets

        if self.sequence_length == 0:
            x_input, file_path = self.get_input(jpg_paths, pcd_paths)
        # Get the input. Dealing with sequences here.
        else:
            count = 0
            x_input, file_path = [], []
            while count != self.sequence_length:
                x, f = self.get_input(jpg_paths, pcd_paths)
                if x is not None and f is not None:
                    x_input.append(x)
                    file_path.append(f)
                    count += 1
            x_input = np.array(x_input)
            file_path = np.array(file_path)
        """
        # Got a proper sample.
        if x_input is not None and y_output is not None and file_path is not None:
            x_inputs.append(x_input)
            y_outputs.append(y_output)
            file_paths.append(file_path)

        assert len(x_inputs) == len(y_outputs)
        assert len(y_outputs) == len(file_paths)
        """
        return x_input, y_output, file_path

    def get_x_input(self):
        pass

    def get_y_output(self):
        pass

    def get_out_filepath(self):
        pass

    def get_input(self, jpg_paths, pcd_paths):
        # Get a random image.
        if self.input_type == "image":
            if len(jpg_paths) == 0:
                print("777")
                return None, None
            jpg_path = random.choice(jpg_paths)
            image = self._load_image(jpg_path)
            file_path = jpg_path
            x_input = image

        # Get a random voxelgrid.
        elif self.input_type == "voxelgrid":
            if len(pcd_paths) == 0:
                return None, None
            pcd_path = random.choice(pcd_paths)
            try:
                voxelgrid = self._load_voxelgrid(pcd_path)
                file_path = pcd_path
                x_input = voxelgrid
            except Exception as e:
                print(e)
                return None, None

        # Get a random pointcloud.
        elif self.input_type == "pointcloud":
            if len(pcd_paths) == 0:
                return None, None
            pcd_path = random.choice(pcd_paths)
            try:
                pointcloud = self._load_pointcloud(pcd_path)
                file_path = pcd_path
                x_input = pointcloud
            except Exception as e:
                print(e)
                return None, None

        # Should not happen.
        else:
            raise Exception("Unknown input_type: " + self.input_type)

        return x_input, file_path
