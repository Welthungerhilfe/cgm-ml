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

import logging
import logging.config
import os
from enum import IntEnum

import numpy as np
import pandas as pd
from cv2 import cv2
from pyntcloud import PyntCloud
from pyntcloud.io import write_ply

from cgm_fusion.calibration import (get_extrinsic_matrix_depth,
                                    get_intrinsic_matrix_depth, get_k_depth)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


HEIGHT = 224
WIDTH = 172


def fuse_point_cloud(points, rgb_vals, confidence, seg_vals):
    df = pd.DataFrame(columns=['x', 'y', 'z', 'red', 'green', 'blue',
                               'seg'])  # 'nx', 'ny', 'nz'])

    df['x'] = points[:, 0]  # saving carthesian coordinates
    df['y'] = points[:, 1]
    df['z'] = points[:, 2]

    df['red'] = rgb_vals[:, 2].astype(np.uint8)  # saving the color
    df['green'] = rgb_vals[:, 1].astype(np.uint8)
    df['blue'] = rgb_vals[:, 0].astype(np.uint8)

    df['c'] = confidence[:].astype(np.float)  # saving the confidence

    df['seg'] = seg_vals[:].astype(np.float)  # saving the segmentation

    # df['nx']    = normals[:, 0]                             # normal x coordinate
    # df['ny']    = normals[:, 1]                             # normal y coordinate
    # df['nz']    = normals[:, 2]                             # normal z coordinate

    new_pc = PyntCloud(df)
    return new_pc


def write_color_ply(fname, points, color_vals, confidence, normals):
    new_pc = fuse_point_cloud(points, color_vals, confidence, normals)
    write_ply(fname, new_pc.points, as_text=True)
    logging.info(fname)


def apply_projection(points, calibration_file):
    intrinsic = get_intrinsic_matrix_depth(calibration_file)

    ext_d = get_extrinsic_matrix_depth(calibration_file, idx=4)

    r_vec = ext_d[:3, :3]
    t_vec = -ext_d[:3, 3]

    k1, k2, k3 = get_k_depth(calibration_file)
    im_coords, _ = cv2.projectPoints(points, r_vec, t_vec, intrinsic[:3, :3],
                                     np.array([k1, k2, 0, 0]))

    return im_coords


class Channel(IntEnum):
    x = 0
    y = 1
    z = 2
    confidence = 3
    red = 4
    green = 5
    blue = 6
    segmentation = 7
    nx = 8
    ny = 9
    nz = 10


def get_depth_channel(ply_path, output_path_np, output_path_png, calibration_file):
    if not os.path.exists(calibration_file):  # check if the califile exists
        logging.error('Calibration does not exist')
        return

    # get a default black image
    nr_of_channels = 1
    viz_image = np.zeros((HEIGHT, WIDTH, nr_of_channels), np.float64)

    try:
        cloud = PyntCloud.from_file(ply_path)  # load the data from the files
    except ValueError as e:
        logging.error("Error reading point cloud")
        logging.error(str(e))
        logging.error(ply_path)

    logging.info(cloud.points.values[:, 3])
    r = cloud.points.values[:, 3]
    logging.info(min(r))
    logging.info(max(r))
    logging.info(np.mean(r))

    points = cloud.points.values[:, :3]  # get x y z
    z = cloud.points.values[:, 2]  # get only z coordinate
    z = (z - min(z)) / (max(z) - min(z))  # normalize the data to 0 to 1

    # iterat of the points and calculat the x y coordinates in the image
    # get the data for calibration
    im_coords = apply_projection(points, calibration_file)

    # manipulate the pixels color value depending on the z coordinate
    # TODO make this a function
    for i, t in enumerate(im_coords):
        x, y = t.squeeze()
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < HEIGHT and y >= 0 and y < WIDTH:
            viz_image[x, y] = z[i]  # 255 #255-255*z[i]

    # img_debug = cv2.normalize(src=viz_image,
    #                           dst=None,
    #                           alpha=0,
    #                           beta=255,
    #                           norm_type=cv2.NORM_MINMAX,
    #                           dtype=cv2.CV_8U)

    cv2.imwrite("/tmp/viz_debug.png", viz_image)

    # resize and  return the image after pricessing
    # dim = (180, 240)
    # viz_image = cv2.resize(viz_image, dim, interpolation = cv2.INTER_AREA)

    #np.save(output_path_np, viz_image)
    #viz_2 = np.load("/tmp/out.npy")

    img_n = cv2.normalize(src=viz_image,
                          dst=None,
                          alpha=0,
                          beta=255,
                          norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_8U)
    cv2.imwrite(output_path_png, img_n)

    return viz_image


def get_rgbd_channel(ply_path, output_path_np, calibration_file):
    if not os.path.exists(calibration_file):  # check if the califile exists
        logging.error('Calibration does not exist')
        return

    # get a default black image
    nr_of_channels = 4
    viz_image = np.zeros((HEIGHT, WIDTH, nr_of_channels), np.float64)

    try:
        cloud = PyntCloud.from_file(ply_path)  # load the data from the files
    except ValueError as e:
        logging.error(" Error reading point cloud ")
        logging.error(str(e))
        logging.error(ply_path)

    points = cloud.points.values[:, :3]  # get x y z
    z = cloud.points.values[:, 2]  # get only z coordinate
    z = (z - min(z)) / (max(z) - min(z))  # normalize the data to 0 to 1
    r = cloud.points.values[:, 4]
    g = cloud.points.values[:, 5]
    b = cloud.points.values[:, 6]

    # iterat of the points and calculat the x y coordinates in the image
    # get the data for calibration
    im_coords = apply_projection(points, calibration_file)
    #im_coords=np.nan_to_num(im_coords)                     #removing nans

    # manipulate the pixels color value depending on the z coordinate
    # TODO make this a function

    for i, t in enumerate(im_coords):
        x, y = t.squeeze()
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < HEIGHT and y >= 0 and y < WIDTH:
            viz_image[x, y, 0] = r[i]
            viz_image[x, y, 1] = g[i]
            viz_image[x, y, 2] = b[i]
            viz_image[x, y, 3] = z[i]

    np.save(output_path_np, viz_image)
    return viz_image


def get_all_channel(ply_path, output_path_np, calibration_file):
    if not os.path.exists(calibration_file):  # check if the califile exists
        logging.error('Calibration does not exist')
        return

    # get a default black image
    nr_of_channels = 11
    viz_image = np.zeros((HEIGHT, WIDTH, nr_of_channels), np.float64)

    try:
        cloud = PyntCloud.from_file(ply_path)  # load the data from the files
    except ValueError as e:
        logging.error(" Error reading point cloud ")
        logging.error(str(e))
        logging.error(ply_path)

    points = cloud.points.values[:, :3]  # get x y z
    x = cloud.points.values[:, 0]
    y = cloud.points.values[:, 1]
    z = cloud.points.values[:, 2]  # get only z coordinate
    z = (z - min(z)) / (max(z) - min(z))  # normalize the data to 0 to 1

    r = cloud.points.values[:, 4]
    g = cloud.points.values[:, 5]
    b = cloud.points.values[:, 6]

    seg = cloud.points.values[:, 7]
    conf = cloud.points.values[:, 3]

    # nx = cloud.points.values[:, 8]
    # nx = cloud.points.values[:, 9]
    # nz = cloud.points.values[:, 10]

    # iterat of the points and calculat the x y coordinates in the image
    # get the data for calibration
    im_coords = apply_projection(points)

    # manipulate the pixels color value depending on the z coordinate
    # TODO make this a function
    for i, t in enumerate(im_coords):
        x, y = t.squeeze()
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < HEIGHT and y >= 0 and y < WIDTH:
            viz_image[x, y, 0] = x[i]
            viz_image[x, y, 1] = y[i]
            viz_image[x, y, 2] = z[i]

            viz_image[x, y, 3] = r[i]
            viz_image[x, y, 4] = g[i]
            viz_image[x, y, 5] = b[i]

            viz_image[x, y, 6] = seg[i]
            viz_image[x, y, 7] = conf[i]

            # viz_image[x, y, 8] = nx[i]
            # viz_image[x, y, 9] = ny[i]
            # viz_image[x, y, 10] = nz[i]

    np.save(output_path_np, viz_image)
    return viz_image


def get_viz_channel(calibration_file,
                    ply_path,
                    channel=Channel.z,
                    output_path="/tmp/output.png"):

    if not os.path.exists(calibration_file):  # check if the califile exists
        logging.error('Calibration does not exist')
        return

    # get a default black image
    nr_of_channels = 1
    viz_image = np.zeros((HEIGHT, WIDTH, nr_of_channels), np.uint8)

    # get the points from the pointcloud
    try:
        cloud = PyntCloud.from_file(ply_path)  # load the data from the files
    except ValueError as e:
        logging.error(" Error reading point cloud ")
        logging.error(str(e))

    # logging.info(int(channel))

    points = cloud.points.values[:, :3]  # get x y z
    z = cloud.points.values[:, int(channel)]  # get only z coordinate
    z = (z - min(z)) / (max(z) - min(z))  # normalize the data to 0 to 1

    # iterat of the points and calculat the x y coordinates in the image
    # get the data for calibration
    im_coords = apply_projection(points, calibration_file)

    # manipulate the pixels color value depending on the z coordinate
    # TODO make this a function
    for i, t in enumerate(im_coords):
        x, y = t.squeeze()
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < HEIGHT and y >= 0 and y < WIDTH:
            viz_image[x, y] = 255 * z[i]

    # resize and  return the image after pricessing
    #imgScale = 0.25
    #newX, newY = viz_image.shape[1] * imgScale, viz_image.shape[0] * imgScale
    cv2.imwrite(output_path, viz_image)

    return viz_image


def get_viz_rgb(ply_path):
    '''Function to get the rgb from a point cloud as an image for visualization'''
    get_viz_channel(ply_path, channel=Channel.red, output_path="/tmp/red.png")


def get_viz_confidence(ply_path):
    '''Function to get the confidence from a point cloud as an image for visualization'''
    get_viz_channel(ply_path,
                    channel=Channel.confidence,
                    output_path="/tmp/confidence.png")


def get_viz_depth(ply_path):
    get_viz_channel(ply_path, channel=Channel.z, output_path="/tmp/depth.png")


def get_viz_segmentation(ply_path):
    '''Function to get the segmentation from a point cloud as an image for visualization'''
    get_viz_channel(ply_path,
                    channel=Channel.segmentation,
                    output_path="/tmp/segmentation.png")


def get_viz_normal_z(ply_path):
    get_viz_channel(ply_path,
                    channel=Channel.nz,
                    output_path="/tmp/normals.png")
