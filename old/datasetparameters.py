'''
This is a pool of different dataset-parameters. Used for preprocessing/training.
'''

random_seed = 667

# For creating pointclouds.
dataset_parameters_pointclouds = {}
dataset_parameters_pointclouds["input_type"] = "pointcloud"
dataset_parameters_pointclouds["output_targets"] = ["height"]
dataset_parameters_pointclouds["random_seed"] = random_seed
dataset_parameters_pointclouds["pointcloud_target_size"] = 10000
dataset_parameters_pointclouds["pointcloud_random_rotation"] = False
dataset_parameters_pointclouds["sequence_length"] = 0

# For creating voxelgrids.
dataset_parameters_voxelgrids = {}
dataset_parameters_voxelgrids["input_type"] = "voxelgrid"
dataset_parameters_voxelgrids["output_targets"] = ["height"]
dataset_parameters_voxelgrids["random_seed"] = random_seed
dataset_parameters_voxelgrids["voxelgrid_target_shape"] = (32, 32, 32)
dataset_parameters_voxelgrids["voxel_size_meters"] = 0.1
dataset_parameters_voxelgrids["voxelgrid_random_rotation"] = False
dataset_parameters_voxelgrids["sequence_length"] = 0
