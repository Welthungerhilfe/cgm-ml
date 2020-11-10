# examples/Python/Basic/pointcloud.py

import numpy as np
import open3d

ENABLE_VISUALIZATION = False
DOWNSAMPLE = True

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = open3d.io.read_point_cloud("/data/home/cpfitzner/test.pcd")
    print(pcd)
    print(np.asarray(pcd.points))
    if ENABLE_VISUALIZATION:
        open3d.visualization.draw_geometries([pcd])

    downpcd = pcd
    if DOWNSAMPLE:
        print("DOWNSAMPLE the point cloud with a voxel of 0.05")
        downpcd = pcd.voxel_down_sample(voxel_size=0.05)
        print(downpcd)
        print(np.asarray(downpcd.points))

    if ENABLE_VISUALIZATION:
        open3d.visualization.draw_geometries([downpcd])

    print("Recompute the normal of the DOWNSAMPLEd point cloud")
    downpcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    if ENABLE_VISUALIZATION:
        open3d.visualization.draw_geometries([downpcd])

    print("Print a normal vector of the 0th point")
    print(downpcd.normals[0])
    print("Print the normal vectors of the first 10 points")
    print(np.asarray(downpcd.normals)[:10, :])
    print("x: ")
    print(np.asarray(downpcd.normals)[0, 0])
    print("y: ")
    print(np.asarray(downpcd.normals)[0, 1])
    print("z: ")
    print(np.asarray(downpcd.normals)[0, 2])
    print("")

    print("Load a polygon volume and use it to crop the original point cloud")
    vol = open3d.visualization.read_selection_polygon_volume(
        "../../TestData/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    if ENABLE_VISUALIZATION:
        open3d.visualization.draw_geometries([chair])
        print("")

        print("Paint chair")
        chair.paint_uniform_color([1, 0.706, 0])
        open3d.visualization.draw_geometries([chair])
    print("")
