#
# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Dr. Christian Pfitzner <christian.pfitzner@th-nuernberg.de> for Welthungerhilfe
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


'''
import the necessary functions
'''
import sys
import os  
import cv2
import numpy as np
import pandas as pd
import math
from math import sin,cos,radians

import open3d as o3d
import logging
import matplotlib.pyplot as plt
from PIL import Image

from cgm_fusion.calibration import get_intrinsic_matrix, get_extrinsic_matrix, get_k
from cgm_fusion.utility import write_color_ply, fuse_point_cloud , get_depth_channel

from pyntcloud import PyntCloud

def projectPoints(pcd_points,calibration_file):

    #get the data for calibration
    intrinsic  = get_intrinsic_matrix(calibration_file)
    ext_d      = get_extrinsic_matrix(calibration_file,idx=4)

    r_vec      = ext_d[:3, :3]
    t_vec      = -ext_d[:3, 3]

    k1, k2, k3 = get_k(calibration_file)

    im_coords, _ = cv2.projectPoints(pcd_points, r_vec, t_vec, intrinsic[:3, :3], np.array([k1, k2, 0, 0]))

    return im_coords,_

def get_depth_image_from_point_cloud(calibration_file, pcd_file, output_file):
    if not os.path.exists(pcd_file):                        # check all files exist
        logging.error ('Point cloud does not exist')
        return


    if not os.path.exists(calibration_file):                # check if the califile exists
        logging.error ('Calibration does not exist')
        return 


    try:
        cloud      = PyntCloud.from_file(pcd_file)         # load the data from the files
    except ValueError:
        logging.error(" Error reading point cloud ")
        raise
        return


    # points       = cloud.points.values[:, :3]
    z = cloud.points.values[:, 3]

    print (cloud.points.values.shape)

    height = 172                                                            # todo: get this from calibration file
    width  = 224

    z      = (z - min(z)) / (max(z) - min(z))                  # normalize the data to 0 to 1

    # print (z)

    # print (z.size)

    # iterat of the points and calculat the x y coordinates in the image
    # get the data for calibration 
    # im_coords = apply_projection(points)

    # manipulate the pixels color value depending on the z coordinate
    # TODO make this a function
    # for i, t in enumerate(im_coords):
    #     x, y = t.squeeze()
    #     x = int(np.round(x))
    #     y = int(np.round(y))
    #     if x >= 0 and x < width and y >= 0 and y < height:
    #         viz_image[x,y] = 255*z[i]

    # # resize and  return the image after pricessing
    # imgScale  = 0.25
    # newX,newY = viz_image.shape[1]*imgScale, viz_image.shape[0]*imgScale
    # cv2.imwrite('/tmp/depth_visualization.png', viz_image) 



#111 utilit get vizc_channel


    depth_img = np.resize(z*255, [224, 172,  3])

    depth_img_resize = cv2.resize(z*255, (180, 180 ))                  # todo: make width and height variable

    #cv2.imwrite("/tmp/depth_224x172.png", depth_img)
    #cv2.imwrite("/tmp/depth_240x180.png", depth_img_resize)


    # not sure if we need this 
    
    # # get the data for calibration
    # intrinsic  = get_intrinsic_matrix()
    # ext_d      = get_extrinsic_matrix(4)

    # r_vec      = ext_d[:3, :3]
    # t_vec      = -ext_d[:3, 3]

    # k1, k2, k3 = get_k()
    # im_coords, _ = cv2.projectPoints(points, r_vec, t_vec, intrinsic[:3, :3], np.array([k1, k2, 0, 0]))



def fuse_rgbd(calibration_file,pcd_file,image, seg_path=0): #TODO:uncomment zero for segmentation fusion
    
    try:
        cloud      = PyntCloud.from_file(pcd_file)         # load the data from the files
    except ValueError:
        logging.error(" Error reading point cloud ")
        raise

    points     = cloud.points.values[:, :3]
    #getting projected points
    im_coords, _ = projectPoints(points,calibration_file)
    
    #setting the RGB image dimensions
    scale = 0.1
    width = int(1920 * scale)
    height = int(1080 * scale)
    #reading image and resizing
    pil_im = image.resize((width, height), Image.ANTIALIAS)
    im_array = np.asarray(pil_im)

    #TODO:comment out for segmentation
    # pil_im2 = Image.open(seg_path)
    # pil_im2 = pil_im2.resize((height, width), Image.ANTIALIAS)
    # im_array2 = np.asarray(pil_im2)
    
    #initialize an empty black image
    viz_image = np.zeros((width, height, 3))
    
    #addING depth into RGB array
    map = plt.get_cmap('rainbow')
    pcd_name=pcd_file.split("/")[-1]

    for i in range(len(points)):
        x = int(im_coords[i][0][0] * scale)
        y = int(im_coords[i][0][1] * scale)
        if x >= 0 and y >= 0 and x < width and y < height:
            #segm = im_array2[y][x][1] / 255.0
            depth = points[i][2]
            color = map(depth - 0.75)

            if '_100_' in pcd_name or '_101_' in pcd_name or '_102_' in pcd_name:
                newx = y - 1        
                newy = x - 1
            elif '_200_' in pcd_name or '_201_' in pcd_name or '_202_' in pcd_name:
                newx = height - y - 1
                newy = width - x - 1

            viz_image[newy][newx][0] = color[0]
            viz_image[newy][newx][1] = color[1]
            viz_image[newy][newx][2] = color[2] 
            #viz_image[x][y][3]= segm
    
    return viz_image


def apply_fusion(calibration_file, pcd_file, jpg_file, seg_path):
    ''' 
    check the path if everything is correct
    '''
    if not os.path.exists(pcd_file):                        # check all files exist
        logging.error ('Point cloud does not exist')
        return
        
    if not os.path.exists(jpg_file):                        # check if the jpg file exists
        logging.error ('Image does not exist')
        return 

    if not os.path.exists(seg_path):                        # check if segmentation exists
        logging.error('Segmentation not found')
        return

    if not os.path.exists(calibration_file):                # check if the califile exists
        logging.error ('Calibration does not exist')
        return 


    try:
        cloud      = PyntCloud.from_file(pcd_file)         # load the data from the files
    except ValueError:
        logging.error(" Error reading point cloud ")
        raise
        return

    jpg        = cv2.imread(jpg_file, -1)      
    jpg        = cv2.flip( jpg, 0 )

    seg        = cv2.imread(seg_path, -1)
    seg        = cv2.flip( seg, 0)

    hh, ww, _  = jpg.shape
    
    points     = cloud.points.values[:, :3]
    #np.savetxt("points.txt",points)
    confidence = cloud.points.values[:, 3]
    
    # get the data for calibration
    
    im_coords, _ = projectPoints(points,calibration_file)
    
    color_vals   = np.zeros_like(points)
    
    segment_vals = np.zeros_like(points)

    

    for i, t in enumerate(im_coords):
        x, y = t.squeeze()
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < ww and y >= 0 and y < hh:
            color_vals[i, :]   = jpg[y, x]
            segment_vals[i, :] = seg[y, x] 
            
    


    
    # #convert from pyntcloud to open3d
    # cloud_open3d = o3d.io.read_point_cloud(pcd_file)

    # #calculate the normals from the existing cloud
    # cloud_open3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    fused_point_cloud = fuse_point_cloud(points, color_vals, confidence, segment_vals)#, np.asarray(cloud_open3d.normals))
    
    return  fused_point_cloud





