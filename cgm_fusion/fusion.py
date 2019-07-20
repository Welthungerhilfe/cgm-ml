
import sys
import os  
import cv2
import numpy as np
import pandas as pd


from cgm_fusion.calibration import get_intrinsic_matrix, get_extrinsic_matrix, get_k
from cgm_fusion.utility import write_color_ply, fuse_point_cloud

from pyntcloud import PyntCloud


# def find_corresponding_point_cloud:


# def find_corresponding_image:





def apply_fusion(calibration_file, pcd_file, jpg_file):

    # check all files exist
    if not os.path.exists(pcd_file): 
        print ('Point cloud does not exist')
        return
        
    if not os.path.exists(jpg_file):
        print ('Image does not exist')
        return 

    if not os.path.exists(calibration_file):
        print ('Calibration does not exist')
        return 


    # load the data from the files
    cloud      = PyntCloud.from_file(pcd_file)  
    jpg        = cv2.imread(jpg_file, -1)

    hh, ww, _  = jpg.shape

    points     = cloud.points.values[:, :3]


    # get the data for calibration
    intrinsic  = get_intrinsic_matrix()
    ext_d      = get_extrinsic_matrix(4)
    ext_rgb    = get_extrinsic_matrix(3)
    diff       = ext_rgb @ np.linalg.inv(ext_d)


    Hpoints    = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    Hpoints    = np.transpose(diff @ Hpoints.T)

    im_coords  = np.transpose(intrinsic @ Hpoints.T)
    color_vals = np.zeros_like(points)

    for i, t in enumerate(im_coords):
        x, y, _ = t
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < ww and y >= 0 and y < hh:
            color_vals[i, :] = jpg[y, x]

    fused_point_cloud = fuse_point_cloud(points,color_vals)     
            
    write_color_ply('color_pc.ply', points, color_vals)
    # return  fused_point_cloud



    r_vec = ext_d[:3, :3]
    t_vec = -ext_d[:3, 3]

    k1, k2, k3 = get_k()
    im_coords, _ = cv2.projectPoints(points, r_vec, t_vec, intrinsic[:3, :3], np.array([k1, k2, 0, 0]))


    color_vals = np.zeros_like(points)

    for i, t in enumerate(im_coords):
        x, y = t.squeeze()
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < ww and y >= 0 and y < hh:
            color_vals[i, :] = jpg[y, x]

    write_color_ply('color_pc_cv2.ply', points, color_vals)

    fused_point_cloud = fuse_point_cloud(points, color_vals)

    return  fused_point_cloud





