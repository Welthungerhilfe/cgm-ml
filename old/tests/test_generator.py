import unittest
from cgmcore.datagenerator import DataGenerator, get_dataset_path, create_datagenerator_from_parameters


class TestGenerator(unittest.TestCase):

    #@unittest.skip("demonstrating skipping")
    def test_pointcloud_generation(self):
        dataset_path = get_dataset_path()

        dataset_parameters_pointclouds = {}
        dataset_parameters_pointclouds["input_type"] = "pointcloud"
        dataset_parameters_pointclouds["output_targets"] = ["height", "weight"]
        dataset_parameters_pointclouds["random_seed"] = 666
        dataset_parameters_pointclouds["pointcloud_target_size"] = 30000
        dataset_parameters_pointclouds["pointcloud_random_rotation"] = True
        dataset_parameters_pointclouds["dataset_size_train"] = 1000
        dataset_parameters_pointclouds["dataset_size_test"] = 20

        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)
        dataset = next(data_generator.generate(size=1, yield_file_paths=True, verbose=True))
        assert dataset[0].shape == (1, 30000, 4)


    #@unittest.skip("demonstrating skipping")
    def test_voxelgrid_generation(self):
        dataset_path = get_dataset_path()

        dataset_parameters_voxelgrids = {}
        dataset_parameters_voxelgrids["input_type"] = "voxelgrid"
        dataset_parameters_voxelgrids["output_targets"] = ["height", "weight"]
        dataset_parameters_voxelgrids["random_seed"] = 666
        dataset_parameters_voxelgrids["voxelgrid_target_shape"] = (32, 32, 32)
        dataset_parameters_voxelgrids["voxel_size_meters"] = 0.1
        dataset_parameters_voxelgrids["voxelgrid_random_rotation"] = True
        dataset_parameters_voxelgrids["dataset_size_train"] = 6000
        dataset_parameters_voxelgrids["dataset_size_test"] = 1000

        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_voxelgrids)
        dataset = next(data_generator.generate(size=1, yield_file_paths=True, verbose=True))
        assert dataset[0].shape == (1, 32, 32, 32)


    def test_voxelgrid_generation_with_sequences(self):
        dataset_path = get_dataset_path()

        dataset_parameters_voxelgrids = {}
        dataset_parameters_voxelgrids["input_type"] = "voxelgrid"
        dataset_parameters_voxelgrids["output_targets"] = ["height", "weight"]
        dataset_parameters_voxelgrids["random_seed"] = 666
        dataset_parameters_voxelgrids["voxelgrid_target_shape"] = (32, 32, 32)
        dataset_parameters_voxelgrids["voxel_size_meters"] = 0.1
        dataset_parameters_voxelgrids["voxelgrid_random_rotation"] = True
        dataset_parameters_voxelgrids["dataset_size_train"] = 6000
        dataset_parameters_voxelgrids["dataset_size_test"] = 1000
        dataset_parameters_voxelgrids["sequence_length"] = 8

        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_voxelgrids)
        dataset = next(data_generator.generate(size=1, yield_file_paths=True, verbose=True))
        assert dataset[0].shape == (1, 8, 32, 32, 32)


if __name__ == '__main__':
    unittest.main()
