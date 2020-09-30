import random
from pyntcloud import PyntCloud
import numpy as np
import logging
import etl.utils as etl_utils

log = logging.getLogger(__name__)


class PCDataLoader:
    def __init__(self, config):
        self.pointcloud_cache = {}
        self.sequence_length = config.getint('pointcloud', 'sequence_length')
        self.pointcloud_target_size = config.getint('pointcloud',
                                                    'pointcloud_target_size')
        self.pointcloud_random_rotation = config.getboolean(
            'pointcloud', 'pointcloud_random_rotation')

    def _rotate_point_cloud(self, point_cloud):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0], [-sinval, cosval, 0],
                                    [0, 0, 1]])

        rotated_data = np.zeros(point_cloud.shape, dtype=np.float32)
        for k in range(point_cloud.shape[0]):

            shape_pc = point_cloud[k, ...]
            rotated_data[k, ...] = np.dot(
                shape_pc.reshape((-1, 3)), rotation_matrix)

        return rotated_data

    def _load_pointcloud(self, pcd_path, preprocess=True, augmentation=True):
        point_cloud = self.pointcloud_cache.get(pcd_path, None)
        if point_cloud is None:
            point_cloud = PyntCloud.from_file(pcd_path).points.values
            point_cloud = np.array(point_cloud)

            if self.pointcloud_target_size is not None and preprocess is True:
                point_cloud = point_cloud[:self.pointcloud_target_size]
                if len(point_cloud) < self.pointcloud_target_size:
                    zeros = np.zeros(
                        (self.pointcloud_target_size - len(point_cloud), 4))
                    point_cloud = np.concatenate([point_cloud, zeros])

            if self.pointcloud_random_rotation is True and augmentation is True:
                numpy_points = point_cloud[:, 0:3]
                numpy_points = self._rotate_point_cloud(numpy_points)
                point_cloud[:, 0:3] = numpy_points

        return point_cloud

    def load_data(self, jpg_paths, pcd_paths):
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

        return x_input, file_path

    def get_input(self, jpg_paths, pcd_paths):
        if len(pcd_paths) == 0:
            return None, None
        pcd_path = random.choice(pcd_paths)
        try:
            pointcloud = self._load_pointcloud(pcd_path)
            file_path = pcd_path
            x_input = pointcloud
        except Exception as e:
            log.exception("Error in getting point cloud input")
            return None, None
        return x_input, file_path


class VoxelDataLoader:
    def __init__(self, config):
        self.sequence_length = config.getint('voxelgrid', 'sequence_length')
        self.voxelgrid_random_rotation = config.getboolean(
            'voxelgrid', 'voxelgrid_random_rotation')
        self.voxel_size_meters = config.getfloat('voxelgrid',
                                                 'voxel_size_meters')
        voxelgrid_target_shape = config.get('voxelgrid',
                                            'voxelgrid_target_shape')
        # TODO: improve
        self.voxelgrid_target_shape = tuple(
            [int(x) for x in voxelgrid_target_shape.split(',')])

    def _load_voxelgrid(self, pcd_path, preprocess=True, augmentation=True):
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

        voxelgrid = point_cloud.structures[voxelgrid_id].get_feature_vector(
            mode="density")

        # Do the preprocessing.
        if preprocess:
            voxelgrid = etl_utils.ensure_voxelgrid_shape(
                voxelgrid, self.voxelgrid_target_shape)
            assert voxelgrid.shape == self.voxelgrid_target_shape

        return voxelgrid

    def _rotate_point_cloud(self, point_cloud):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0], [-sinval, cosval, 0],
                                    [0, 0, 1]])

        rotated_data = np.zeros(point_cloud.shape, dtype=np.float32)
        for k in range(point_cloud.shape[0]):

            shape_pc = point_cloud[k, ...]
            rotated_data[k, ...] = np.dot(
                shape_pc.reshape((-1, 3)), rotation_matrix)

        return rotated_data

    def get_input(self, jpg_paths, pcd_paths):
        if len(pcd_paths) == 0:
            return None, None
        pcd_path = random.choice(pcd_paths)
        try:
            voxelgrid = self._load_voxelgrid(pcd_path)
            file_path = pcd_path
            x_input = voxelgrid
        except Exception as e:
            log.exception("Error in getting voxelgrid input")
            return None, None

        return x_input, file_path

    def load_data(self, jpg_paths, pcd_paths):
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

        return x_input, file_path


class DataLoaderFactory(object):
    def factory(input_type, **kwargs):
        if input_type == 'pointcloud':
            return PCDataLoader(**kwargs)
        if input_type == 'voxelgrid':
            return VoxelDataLoader(**kwargs)
        raise Exception("Unknown input_type %s" % str(input_type))

    factory = staticmethod(factory)
