import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")
import unittest
from cgmcore.preprocesseddatagenerator import get_dataset_path, create_datagenerator_from_parameters
import time
import pickle


class TestPreprocessedDataGenerator(unittest.TestCase):

    @unittest.skip("demonstrating skipping")
    def test_pointcloud_measuring_time(self):
        dataset_path = get_dataset_path()
        print("Using dataset path:", dataset_path)

        dataset_parameters_pointclouds = {}
        dataset_parameters_pointclouds["input_type"] = "pointcloud"
        dataset_parameters_pointclouds["random_seed"] = 666
        dataset_parameters_pointclouds["pointcloud_target_size"] = 30000
        dataset_parameters_pointclouds["pointcloud_random_rotation"] = True
        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)
        
        start_time = time.time()
        pointclouds_count = 0
        for qrcode, paths in data_generator.qrcodes_dictionary.items():
            for path in paths:
                with open(path, "rb") as file:
                    (pointcloud, targets) = pickle.load(file)
                    pointclouds_count += 1
                    del pointcloud
                    del targets
        elapsed_time = time.time() - start_time
        print("Loaded {} pointclouds in {} seconds".format(pointclouds_count, elapsed_time))
        
        
    @unittest.skip("demonstrating skipping")
    def test_pointcloud_generation(self):
        dataset_path = get_dataset_path()
        print("Using dataset path:", dataset_path)

        dataset_parameters_pointclouds = {}
        dataset_parameters_pointclouds["input_type"] = "pointcloud"
        dataset_parameters_pointclouds["random_seed"] = 666
        dataset_parameters_pointclouds["pointcloud_target_size"] = 3000
        dataset_parameters_pointclouds["pointcloud_random_rotation"] = True
        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)

        data_generator.analyze_files()

        dataset = next(data_generator.generate(size=100, verbose=True))
        assert dataset[0].shape == (100, 3000, 4), str(dataset[0].shape)
        assert dataset[1].shape == (100, 1), str(dataset[1].shape)
        
    @unittest.skip("demonstrating skipping")    
    def test_voxelgrid_generation(self):
        dataset_path = get_dataset_path()
        print("Using dataset path:", dataset_path)

        dataset_parameters_voxelgrids = {}
        dataset_parameters_voxelgrids["input_type"] = "voxelgrid"
        dataset_parameters_voxelgrids["random_seed"] = 666
        dataset_parameters_voxelgrids["voxelgrid_target_shape"] = (32, 32, 32)
        dataset_parameters_voxelgrids["voxel_size_meters"] = 0.1
        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_voxelgrids)

        data_generator.analyze_files()

        dataset = next(data_generator.generate(size=100, verbose=True))
        assert dataset[0].shape == (100, 3000, 4), str(dataset[0].shape)
        assert dataset[1].shape == (100, 1), str(dataset[1].shape)

    @unittest.skip("demonstrating skipping")  
    def test_filter(self):
        dataset_path = get_dataset_path("../../data/preprocessed")
        print("Using dataset path:", dataset_path)

        dataset_parameters_pointclouds = {}
        dataset_parameters_pointclouds["input_type"] = "pointcloud"
        dataset_parameters_pointclouds["random_seed"] = 666
        dataset_parameters_pointclouds["pointcloud_target_size"] = 3000
        dataset_parameters_pointclouds["pointcloud_random_rotation"] = True
        dataset_parameters_pointclouds["filter"] = "front"
        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)
        data_generator.analyze_files()

        dataset = next(data_generator.generate(size=100, verbose=True))
        assert dataset[0].shape == (100, 3000, 3), str(dataset[0].shape)
        assert dataset[1].shape == (100, 1), str(dataset[1].shape)
        
        dataset_parameters_pointclouds = {}
        dataset_parameters_pointclouds["input_type"] = "pointcloud"
        dataset_parameters_pointclouds["random_seed"] = 666
        dataset_parameters_pointclouds["pointcloud_target_size"] = 3000
        dataset_parameters_pointclouds["pointcloud_random_rotation"] = True
        dataset_parameters_pointclouds["filter"] = "360"
        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)
        data_generator.analyze_files()

        dataset = next(data_generator.generate(size=100, verbose=True))
        assert dataset[0].shape == (100, 3000, 3), str(dataset[0].shape)
        assert dataset[1].shape == (100, 1), str(dataset[1].shape)
        
        dataset_parameters_pointclouds = {}
        dataset_parameters_pointclouds["input_type"] = "pointcloud"
        dataset_parameters_pointclouds["random_seed"] = 666
        dataset_parameters_pointclouds["pointcloud_target_size"] = 3000
        dataset_parameters_pointclouds["pointcloud_random_rotation"] = True
        dataset_parameters_pointclouds["filter"] = "back"
        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)
        data_generator.analyze_files()

        dataset = next(data_generator.generate(size=100, verbose=True))
        assert dataset[0].shape == (100, 3000, 3), str(dataset[0].shape)
        assert dataset[1].shape == (100, 1), str(dataset[1].shape)

    @unittest.skip("demonstrating skipping")      
    def test_sequence_pointcloud(self):
        dataset_path = get_dataset_path("../../data/preprocessed")
        print("Using dataset path:", dataset_path)

        dataset_parameters_pointclouds = {}
        dataset_parameters_pointclouds["input_type"] = "pointcloud"
        dataset_parameters_pointclouds["random_seed"] = 666
        dataset_parameters_pointclouds["pointcloud_target_size"] = 3000
        dataset_parameters_pointclouds["pointcloud_random_rotation"] = True
        dataset_parameters_pointclouds["filter"] = "360"
        dataset_parameters_pointclouds["sequence_length"] = 8
        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)
        data_generator.analyze_files()

        dataset = next(data_generator.generate(size=10, verbose=True))
        assert dataset[0].shape == (10, 8, 3000, 3), str(dataset[0].shape)
        assert dataset[1].shape == (10, 1), str(dataset[1].shape)
        
    #@unittest.skip("demonstrating skipping")
    def test_sequence_rgb_map(self):
        dataset_path = get_dataset_path("../../data/preprocessed")
        print("Using dataset path:", dataset_path)

        dataset_parameters_rgbmaps = {}
        dataset_parameters_rgbmaps["input_type"] = "rgbmap"
        dataset_parameters_rgbmaps["output_targets"] = ["weight"]
        dataset_parameters_rgbmaps["random_seed"] = 666
        dataset_parameters_rgbmaps["filter"] = "360"
        dataset_parameters_rgbmaps["sequence_length"] = 4
        dataset_parameters_rgbmaps["rgbmap_target_width"] = 64
        dataset_parameters_rgbmaps["rgbmap_target_height"] = 64
        dataset_parameters_rgbmaps["rgbmap_scale_factor"] = 1.0
        dataset_parameters_rgbmaps["rgbmap_axis"] = "horizontal"
        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_rgbmaps)
        data_generator.analyze_files()

        generator = data_generator.generate(size=10, verbose=False)
        for _ in range(10):
            dataset = next(generator)
            assert dataset[0].shape == (10, 4, 64, 64, 3), str(dataset[0].shape)
            assert dataset[1].shape == (10, 1), str(dataset[1].shape)
            print(dataset[1])
        
    @unittest.skip("demonstrating skipping")
    def test_sequence_rgb_map_stress(self):
        dataset_path = get_dataset_path("../../data/preprocessed")
        print("Using dataset path:", dataset_path)

        dataset_parameters_rgbmaps = {}
        dataset_parameters_rgbmaps["input_type"] = "rgbmap"
        dataset_parameters_rgbmaps["random_seed"] = 666
        dataset_parameters_rgbmaps["filter"] = "360"
        dataset_parameters_rgbmaps["sequence_length"] = 8
        dataset_parameters_rgbmaps["rgbmap_target_width"] = 64
        dataset_parameters_rgbmaps["rgbmap_target_height"] = 64
        dataset_parameters_rgbmaps["rgbmap_scale_factor"] = 1.0
        dataset_parameters_rgbmaps["rgbmap_axis"] = "vertical"
        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_rgbmaps)
        data_generator.analyze_files()

        dataset = next(data_generator.generate(size=10000, verbose=True))
        assert dataset[0].shape == (10000, 64, 64, 3), str(dataset[0].shape)
        assert dataset[1].shape == (10000, 1), str(dataset[1].shape)
    
    
if __name__ == '__main__':
    unittest.main()
