# examples/Python/Basic/pointcloud.py

import logging
import logging.config
import numpy as np
import open3d

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENABLE_VISUALIZATION = False
DOWNSAMPLE = True

if __name__ == "__main__":

    logger.info("Load a ply point cloud, print it, and render it")
    pcd = open3d.io.read_point_cloud("/data/home/cpfitzner/test.pcd")
    logger.info(pcd)
    logger.info(np.asarray(pcd.points))
    if ENABLE_VISUALIZATION:
        open3d.visualization.draw_geometries([pcd])

    downpcd = pcd
    if DOWNSAMPLE:
        logger.info("DOWNSAMPLE the point cloud with a voxel of 0.05")
        downpcd = pcd.voxel_down_sample(voxel_size=0.05)
        logger.info(downpcd)
        logger.info(np.asarray(downpcd.points))

    if ENABLE_VISUALIZATION:
        open3d.visualization.draw_geometries([downpcd])

    logger.info("Recompute the normal of the DOWNSAMPLEd point cloud")
    downpcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    if ENABLE_VISUALIZATION:
        open3d.visualization.draw_geometries([downpcd])

    logger.info("Print a normal vector of the 0th point")
    logger.info(downpcd.normals[0])
    logger.info("Print the normal vectors of the first 10 points")
    logger.info(np.asarray(downpcd.normals)[:10, :])
    logger.info("x: ")
    logger.info(np.asarray(downpcd.normals)[0, 0])
    logger.info("y: ")
    logger.info(np.asarray(downpcd.normals)[0, 1])
    logger.info("z: ")
    logger.info(np.asarray(downpcd.normals)[0, 2])
    logger.info("")

    logger.info("Load a polygon volume and use it to crop the original point cloud")
    vol = open3d.visualization.read_selection_polygon_volume(
        "../../TestData/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    if ENABLE_VISUALIZATION:
        open3d.visualization.draw_geometries([chair])
        logger.info("")

        logger.info("Paint chair")
        chair.paint_uniform_color([1, 0.706, 0])
        open3d.visualization.draw_geometries([chair])
    logger.info("")
