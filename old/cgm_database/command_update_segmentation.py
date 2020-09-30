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

from PIL import Image

from io import BytesIO

#import the neccessary packages for the sensor fusion
import cgm_fusion.utility
import cgm_fusion.calibration 
from cgm_fusion.fusion import apply_fusion 

# import core packages from cgm
from cgmcore.utils import load_pcd_as_ndarray
from cgmcore import  utils

# import packages for visualizationi 
from pyntcloud import PyntCloud
from timeit import default_timer as timer


import datetime



import tensorflow as tf

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












# load model for segmentation
modelType = "/whhdata/models/segmentation/xception_model"

MODEL = DeepLabModel(modelType)
print('model loaded successfully : ' + modelType)






def update_qrs(unique_qr_codes, process_index):
#def update_qrs(unique_qr_codes):

    # initialize the rrogress bar with the maxium number of unique qr codes
    #bar = progressbar.ProgressBar(max_value=len(unique_qr_codes))
    qr_counter = 0

#    for qr in tqdm(unique_qr_codes,position=process_index):
    for qr in unique_qr_codes:
        
        qr_counter = qr_counter + 1
        logging.error(qr_counter)
        logging.error(qr)

        # exclude the folling qrcodes
        if qr == "{qrcode}":
            continue
        if qr == "data":
            continue


        # get all images from a unique qr
        sql_statement  = "SELECT  path, type, tango_timestamp FROM artifact "
        sql_statement += " WHERE qr_code = '{}'".format(qr)
        sql_statement += " AND type = 'rgb'"

        connector1 = dbutils.connect_to_main_database()
        all_rgb    = connector1.execute(sql_statement, fetch_all=True)


        for rgb in all_rgb:
            


            # get path and generate output path from it
            img_path = rgb[0]
            img_path = img_path.replace("whhdata", "localssd")
            seg_path = img_path.replace(".jpg", "_SEG.png")

            # load image from path
            try:
                logging.info("Trying to open : " + img_path)
                jpeg_str   = open(img_path, "rb").read()
                orignal_im = Image.open(BytesIO(jpeg_str))
            except IOError:
                print('Cannot retrieve image. Please check file: ' + img_path)
                continue


            # apply segmentation via pre trained model
            logging.info('running deeplab on image %s...' % img_path)
            resized_im, seg_map = MODEL.run(orignal_im)


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
            # img = img.convert("RGB")
            img = img.convert('RGB').resize(orignal_im.size, Image.ANTIALIAS)
            img.save('/tmp/output.png')


            print(seg_path)
            print(img_path)

            logging.info("saved file to" + seg_path)
            img.save(seg_path)






    #bar.finish()




def main():
    # get a list of all unique qr_codes
    sql_statement = "SELECT DISTINCT artifact.qr_code FROM artifact  ORDER BY qr_code ASC;"
    unique_qr_codes = main_connector.execute(sql_statement, fetch_all=True)

    #print(unique_qr_codes)

    # todo: remove the (1) or (2) backup ?
    unique_qr_codes = [x[0] for x in unique_qr_codes]


    # initialze log file
    logging.basicConfig(filename='/tmp/command_update_segmentation.log',level=logging.DEBUG, format='%(asctime)s %(message)s')


    update_qrs(unique_qr_codes)
    
    # Run this in multiprocess mode.
    #utils.multiprocess(unique_qr_codes, 
    #    process_method=update_qrs, 
    #    process_individial_entries=False, 
    #    pass_process_index=True, 
    #    progressbar=False)
    
    print("Done.")





if __name__ == "__main__":
    main()
