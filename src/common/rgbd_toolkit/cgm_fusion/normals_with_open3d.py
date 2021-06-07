# examples/Python/Basic/pointcloud.py

import logging
import logging.config
import numpy as np
import open3d

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

ENABLE_VISUALIZATION = False
DOWNSAMPLE = True

if __name__ == "__main__":

    logging.info("Load a ply point cloud, print it, and render it")
    pcd = open3d.io.read_point_cloud("/data/home/cpfitzner/test.pcd")
    logging.info(pcd)
    logging.info(np.asarray(pcd.points))
    if ENABLE_VISUALIZATION:
        open3d.visualization.draw_geometries([pcd])

    downpcd = pcd
    if DOWNSAMPLE:
        logging.info("DOWNSAMPLE the point cloud with a voxel of 0.05")
        downpcd = pcd.voxel_down_sample(voxel_size=0.05)
        logging.info(downpcd)
        logging.info(np.asarray(downpcd.points))

    if ENABLE_VISUALIZATION:
        open3d.visualization.draw_geometries([downpcd])

    logging.info("Recompute the normal of the DOWNSAMPLEd point cloud")
    downpcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    if ENABLE_VISUALIZATION:
        open3d.visualization.draw_geometries([downpcd])

    logging.info("Print a normal vector of the 0th point")
    logging.info(downpcd.normals[0])
    logging.info("Print the normal vectors of the first 10 points")
    logging.info(np.asarray(downpcd.normals)[:10, :])
    logging.info("x: ")
    logging.info(np.asarray(downpcd.normals)[0, 0])
    logging.info("y: ")
    logging.info(np.asarray(downpcd.normals)[0, 1])
    logging.info("z: ")
    logging.info(np.asarray(downpcd.normals)[0, 2])
    logging.info("")

    logging.info("Load a polygon volume and use it to crop the original point cloud")
    vol = open3d.visualization.read_selection_polygon_volume(
        "../../TestData/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    if ENABLE_VISUALIZATION:
        open3d.visualization.draw_geometries([chair])
        logging.info("")

        logging.info("Paint chair")
        chair.paint_uniform_color([1, 0.706, 0])
        open3d.visualization.draw_geometries([chair])
    logging.info("")
