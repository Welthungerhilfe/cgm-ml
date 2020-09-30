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
import os  
import sys
sys.path.insert(0, "..")

import numpy as np
from numpy import size
import dbutils
import progressbar #@todo remove this
import logging
from tqdm import tqdm


#import the neccessary packages for the sensor fusion
import cgm_fusion.utility
import cgm_fusion.calibration 
from cgm_fusion.fusion import apply_fusion 

# import command_update_segmentation

# import core packages from cgm
from cgmcore.utils import load_pcd_as_ndarray
from cgmcore import  utils

# import packages for visualizationi 
from pyntcloud import PyntCloud
from timeit import default_timer as timer


import tensorflow as tf
from PIL import Image
from io import BytesIO
import datetime


# Connect to database.
main_connector = dbutils.connect_to_main_database()




class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read()) 

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
#    session = tf.Session(config=config, ...)

    self.sess = tf.Session(graph=self.graph, config=config)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    start = datetime.datetime.now()

    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]

    end = datetime.datetime.now()

    diff = end - start
    logging.info("Time taken to evaluate segmentation is : " + str(diff))

    return resized_image, seg_map













''' 
    return the normalized timestamps with values between 0 and 1
'''
def get_timestamps_from_rgb(qr_code):
    connector1 = dbutils.connect_to_main_database()

    # get all artifacts of a certain unique qr code
    sql_statement  = "SELECT  path, type, tango_timestamp FROM artifact "
    sql_statement += " WHERE qr_code = '{}'".format(qr_code)
    sql_statement += " AND type = 'rgb'"

    all_rgb = connector1.execute(sql_statement, fetch_all=True)
    
    timestamps = [x[2] for x in all_rgb]
    path       = [x[0] for x in all_rgb]

    
    if( len(timestamps) == 0): 
        error = np.array([])
        return [error, path]
    
    timestamps      = np.asarray(timestamps)
    return [timestamps, path]



''' 
    import the timestamp from the pcd file from the header
'''
def get_timestamp_from_pcd(pcd_path): 
    filename  = str(pcd_path[0])
    infile    = open(filename, 'r')
    firstLine = infile.readline()

    # get the time from the header of the pcd file
    import re
    timestamp = re.findall("\d+\.\d+", firstLine)

    # check if a timestamp is parsed from the header of the pcd file
    try: 
        return_timestamp = float(timestamp[0])
    except IndexError:
        return_timestamp = []

    return return_timestamp  # index error? IndexError




''' 
    get the timestamps out of the pcd files
'''
def get_timestamps_from_pcd(qr_code): 
    connector2 = dbutils.connect_to_main_database()

    sql_statement  = "SELECT  path FROM artifact "
    sql_statement += " WHERE qr_code = '{}'".format(qr_code)
    sql_statement += " AND type = 'pcd'" 
    path = connector2.execute(sql_statement, fetch_all=True)
    timestamps = np.array([])

    #iterate over all paths pointing to pcds
    for p in path: 
        
        try: 
            stamp = get_timestamp_from_pcd(p)
            timestamps = np.append(timestamps, stamp)
        except IndexError: 
            error = np.array([])
            logging.error("Error with timestamp in pcd")
            return [error, path]



    if( len(timestamps) == 0): 
        error = np.array([])
        return [error, path]
    
    return timestamps, path




def apply_segmentation(jpg_path, model):

    # get path and generate output path from it
    seg_path = jpg_path.replace(".jpg", "_SEG.png")

    # load image from path
    try:
        logging.info("Trying to open : " + jpg_path)
        jpeg_str   = open(jpg_path, "rb").read()
        orignal_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        logging.error('Cannot retrieve image. Please check file: ' + jpg_path)
        return


    # apply segmentation via pre trained model
    logging.info('running deeplab on image %s...' % jpg_path)
    resized_im, seg_map = model.run(orignal_im)


    # convert the image into a binary mask
    width, height = resized_im.size
    dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            color = seg_map[y,x]
            if color == 0:
                dummyImg[y,x] = [0, 0, 0, 255]
            else :
                dummyImg[y,x] = [255,255,255,255]

                
    img = Image.fromarray(dummyImg)
    img = img.convert('RGB').resize(orignal_im.size, Image.ANTIALIAS)
    img.save('/tmp/output.png')

    logging.info("saved file to" + seg_path)
    img.save(seg_path)

    return seg_path





''' 
    return the index of the element in a nparray to the target value
'''
def find_closest(A, target):
    #A must be sorted
    idx   = A.searchsorted(target)
    idx   = np.clip(idx, 1, len(A)-1)
    left  = A[idx-1]
    right = A[idx]
    idx  -= target - left < right - target
    return idx




def update_qrs(unique_qr_codes, process_index):
    # initialize the rrogress bar with the maxium number of unique qr codes
    #bar = progressbar.ProgressBar(max_value=len(unique_qr_codes))
    qr_counter = 0


    # load model for segmentation
    modelType = "/whhdata/models/segmentation/xception_model"
    MODEL     = DeepLabModel(modelType)
    logging.info('model loaded successfully : ' + modelType)


    for qr in tqdm(unique_qr_codes,position=process_index):

        qr_counter = qr_counter + 1

        if qr == "{qrcode}":
            continue

        if qr == "data":
            continue


        [norm_rgb_time, rgb_path] = get_timestamps_from_rgb(qr)
        [norm_pcd_time, pcd_path] = get_timestamps_from_pcd(qr)

        # check if a qr code has rgb and pcd, otherwise the previous function returned -1

        if ( size(norm_rgb_time) == 0 ):
            logging.error("wrong size of jpg")
            logging.error("size rgb: " + str(size(norm_rgb_time)))
            continue

        if ( size(norm_pcd_time) == 0 ): 
            logging.error("wrong size of pcd")    
            logging.error("size pcd: " + str(size(norm_pcd_time)))
            continue

        i = 0


        for pcd in norm_pcd_time:
            nn = find_closest(norm_rgb_time, pcd)

            logging.info("timestamp of rgb: " + "{0:.2f}".format(round(pcd,2))               + " with index " + str(i)) # + " path: " + str(pcd_path[i]))
            logging.info("timestamp of jpg: " + "{0:.2f}".format(round(norm_rgb_time[nn],2)) + " with index " + str(nn))# + " path: " + str(rgb_path[nn]))

            # get the original file path 
            path, filename = os.path.split(str(pcd_path[i]))

            pcd_file = pcd_path[i]
            pcd_file = pcd_file[0]
            jpg_file = rgb_path[nn]
            

            # print (png_file)

            # manipulate the file names for the database
            jpg_file = jpg_file.replace("/localssd/", "/localssd/")
            png_file = pcd_file #.replace("/whhdata/", "/localssd/")

            
            # check if a segmentation for the found jpg exists
            seg_path = jpg_file.replace('.jpg', '_SEG.png')
            if not( os.path.exists(seg_path) ):
                logging.debug('applying segmentation')
                seg_path = apply_segmentation(jpg_file, MODEL)

                # check if the path now exists
                if not( os.path.exists(seg_path) ):
                    logging.error('Segmented file does not exist: ' + seg_path)

            i = i+1

            cali_file = '/whhdata/calibration.xml'
            # the poin
            # cloud is fused and additionally the cloud is saved as ply in the same folder
            try: 
                # TODO add the segmented point cloud to the path
                fused_cloud = apply_fusion(cali_file, pcd_file, jpg_file, seg_path)
            except Exception as e: 
                logging.error("Something went wrong. ")
                logging.error(str(e))
                continue

            # now save the new data to the folder
            fused_folder, pc_filename = os.path.split(str(pcd_file))

            pcd_path_old = pcd_file

            # replace the pcd and the pc_ in the path for fused data
            pc_filename = pcd_path_old.replace(".pcd", ".ply")
            pc_filename = pc_filename.replace("pc_",   "pcrgb_");

            # write the data to the new storage
            pc_filename = pc_filename.replace("/whhdata/qrcode/", "/localssd2/qrcode/")


            # check if folder exists
            pc_folder = os.path.dirname(pc_filename)
            if not(os.path.isfile(pc_folder)): 
                logging.info("Folder does not exist for " + str(pc_filename))
                os.makedirs(pc_folder, exist_ok=True)
                logging.info("Created folder " + str(pc_folder))


            logging.info("Going to writing new fused data to: " + pc_filename)
            
            
            try: 
                fused_cloud.to_file(pc_filename)                        # save the fused point cloud    
                fused_cloud.to_file('/tmp/cloud_debug.ply')             # save for debugging
            except AttributeError :
                loggin.error("An error occured -- skipping this file to save ") 
                continue
    #bar.finish()




def main():
    # get a list of all unique qr_codes
    sql_statement   = "SELECT DISTINCT artifact.qr_code FROM artifact  ORDER BY qr_code ASC;"
    unique_qr_codes = main_connector.execute(sql_statement, fetch_all=True)

    # todo: remove the (1) or (2) backup ?
    unique_qr_codes = [x[0] for x in unique_qr_codes]

    # initialze log file
    logging.basicConfig(filename='/tmp/command_update_fusion.log',level=logging.DEBUG, format='%(asctime)s %(message)s')

    # Run this in multiprocess mode.
    utils.multiprocess(unique_qr_codes, 
        process_method              = update_qrs, 
        process_individial_entries  = False, 
        number_of_workers           = 4,
        pass_process_index          = True, 
        progressbar                 = False, 
        disable_gpu                 =True)
    
    print("*** Done ***.")





if __name__ == "__main__":
    main()
