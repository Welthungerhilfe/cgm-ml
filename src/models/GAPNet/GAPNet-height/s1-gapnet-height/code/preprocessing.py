import numpy as np


def preprocess_pointcloud(pointcloud, subsample_size, channels):
    if subsample_size is not None:
        skip = max(1, round(len(pointcloud) / subsample_size))
        pointcloud_skipped = pointcloud[::skip, :]
        result = np.zeros((subsample_size, pointcloud.shape[1]), dtype="float32")
        result[:len(pointcloud_skipped), :] = pointcloud_skipped[:subsample_size]
        pointcloud = result
    if channels is not None:
        pointcloud = pointcloud[:, channels]
    return pointcloud.astype("float32")


def preprocess_targets(targets, targets_indices):
    if targets_indices is not None:
        targets = targets[targets_indices]
    return targets.astype("float32")
