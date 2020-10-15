import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import azureml.core
from azureml.core.dataset import Dataset
import json
import os
from glob import glob
from pathlib import Path
# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)
import sys
sys.path.append('../cgm-ml')
sys.path.append(os.path.dirname(os.getcwd()))
#print("sys.path",sys.path)
import numpy as np
from numpy import size
import pandas as pd
#import progressbar #@todo remove this 
import logging
from tqdm import tqdm
import dbutils

from DeepLabModel import DeepLabModel,apply_segmentation
from get_timestamps import get_timestamps_from_rgb,get_timestamps_from_pcd
import cgm_fusion.utility as utility
from cgm_fusion.utility import Channel
import cgm_fusion.calibration 
from cgm_fusion.fusion import apply_fusion,fuse_rgbd 
import warnings
warnings.filterwarnings("ignore")

# import command_update_segmentation

# import core packages from cgm
from cgmcore.utils import load_pcd_as_ndarray
from cgmcore import  utils


import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

import logging
logging.getLogger('').handlers = []
logging.basicConfig(filename='./RGBD.log',level=logging.DEBUG, format='%(asctime)s %(message)s')

import argparse


def find_closest(A, target):
    #A must be sorted
    idx   = A.searchsorted(target)
    idx   = np.clip(idx, 1, len(A)-1)
    left  = A[idx-1]
    right = A[idx]
    idx  -= target - left < right - target
    return idx

def update_qrs(unique_qr_codes, process_index=0):
    

    print("Processing..")
    qr_counter = 0
    rgb_paths=[]
    pcd_paths=[]
        
        
    # load model for segmentation
    
    modelType = "./xception_model"
    MODEL     = DeepLabModel(modelType)
    logging.info('model loaded successfully : ' + modelType)


    
    # MODEL_NAME = 'xception_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

    # _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    # _MODEL_URLS = {
    #     'mobilenetv2_coco_voctrainaug':
    #         'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    #     'mobilenetv2_coco_voctrainval':
    #         'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    #     'xception_coco_voctrainaug':
    #         'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    #     'xception_coco_voctrainval':
    #         'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    # }
    # _TARBALL_NAME = 'deeplab_model.tar.gz'

    # model_dir = tempfile.mkdtemp()
    # tf.gfile.MakeDirs(model_dir)

    # download_path = os.path.join(model_dir, _TARBALL_NAME)
    # print('downloading model, this might take a while...')
    # urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
    #                 download_path)
    # print('download completed! loading DeepLab model...')

    # MODEL = DeepLabModel(download_path)
    # print('model loaded successfully!')
    
    
    # load model for segmentation
    
    modelType = "./xception_model"
    MODEL     = DeepLabModel(modelType)
    logging.info('model loaded successfully : ' + modelType)


    for qr in tqdm(unique_qr_codes,position=process_index):

        logging.info("reading qr code"+ str(qr))
        for dirname,dirs,qr_paths in os.walk(Path(qr)):
                for file in qr_paths :
                        dir_path=os.path.join(dirname,file)
                        #print("dir_path",dir_path)
                        if file.endswith(".jpg"):
                            rgb_paths.append(dir_path)
                            [norm_rgb_time, rgb_path] = get_timestamps_from_rgb(rgb_paths)
                        if file.endswith(".pcd"):
                            pcd_paths.append(dir_path)
                            [norm_pcd_time, pcd_path] = get_timestamps_from_pcd(pcd_paths)
            
    # check if a qr code has rgb and pcd, otherwise the previous function returned -1
        qr_counter = qr_counter + 1
        if qr == "{qrcode}":
            continue
        if qr == "data":
            continue

        if ( size(norm_rgb_time) == 0 ):
            logging.error("wrong size of jpg")
            print("wrong size of jpg")
            logging.error("size rgb: " + str(size(norm_rgb_time)))
            continue

        if ( size(norm_pcd_time) == 0 ): 
            logging.error("wrong size of pcd")    
            logging.error("size pcd: " + str(size(norm_pcd_time)))
            continue

        i = 0


        for pcd in norm_pcd_time:
            #print("hi")
            
            nn = find_closest(norm_rgb_time, pcd)
            logging.info("timestamp of rgb: " + "{0:.2f}".format(round(pcd,2))               + " with index " + str(i)) # + " path: " + str(pcd_path[i]))
            logging.info("timestamp of jpg: " + "{0:.2f}".format(round(norm_rgb_time[nn],2)) + " with index " + str(nn))# + " path: " + str(rgb_path[nn]))
            
            # get the original file path 
            path, filename = os.path.split(str(pcd_path[i]))

            pcd_file = pcd_path[i]
            jpg_file = rgb_path[nn]

            qr_folder=str(Path(qr)).split("/")[-1]
            seg_folder_,rgbd_folder_=output_paths()


            # check if a segmentation for the found jpg exists
            seg_path_ = jpg_file.replace('.jpg', '_SEG.png')
            seg_path_,seg_file=os.path.split(seg_path_)  
            seg_folder=os.path.join(seg_folder_,qr_folder)

            if not os.path.exists(seg_folder):
                os.mkdir(seg_folder)
            seg_path=os.path.join(seg_folder,seg_file)
            
            if not( os.path.exists(seg_path) ):
            
                logging.debug('applying segmentation')
                seg_path = apply_segmentation(jpg_file,seg_path, MODEL)
                # check if the path now exists
            if not( os.path.exists(seg_path) ):
                    logging.error('Segmented file does not exist: ' + seg_path)

            i = i+1

            calibration_file="./calibration.xml"
            
            try: 
                #fused_cloud = apply_fusion(calibration_file,pcd_file, jpg_file, seg_path)
                rgbdseg_arr = fuse_rgbd(calibration_file,pcd_file, jpg_file, seg_path)
            
                                            
            except Exception as e: 
                logging.error("Something went wrong. ")
                
                logging.error(str(e))
                continue

            # now save the new data to the folder
            fused_folder, pc_filename = os.path.split(str(pcd_file))
            pcd_path_old = pcd_file
            
            # replace the pcd and the pc_ in the path for fused data
            pc_filename = pcd_path_old.replace(".pcd", ".ply")
            pc_filename = pc_filename.replace("pc_",   "pcrgb_")
            
            qr_rgbd=os.path.join(rgbd_folder_,qr_folder)
            
            

            rgbd_filename = pc_filename.replace(fused_folder.split("measure")[0], qr_rgbd+"/")
            

            # write the data to the new storage
            rgbd_filename=rgbd_filename.replace(".ply",".npy")
            rgbd_filename=rgbd_filename.replace("/pc/","/rgbd/")
            
            

            rgbd_folder=os.path.dirname(rgbd_filename)
            if not(os.path.isfile(rgbd_folder)): 
                logging.info("Folder does not exist for " + str(rgbd_filename))
                os.makedirs(rgbd_folder, exist_ok=True)
                logging.info("Created folder " + str(rgbd_folder))


            logging.info("Going to writing new fused data to: " + rgbd_filename)

            np_path="/np/" #dummy
            
            
                
            try: 
                #fused_cloud.to_file(pc_filename)
                #fig=plt.figure() 
                #plt.imshow(rgbdseg_arr)
                #plt.imsave(rgbd_filename,rgbdseg_arr) 
                np.save(rgbd_filename,rgbdseg_arr)
    
            except AttributeError :
                logging.error("An error occured -- skipping this file to save ") 
                continue


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Give in the qrcode folder to get rgbd data')
    parser.add_argument('--input',
                       metavar='inputpath',
                       type=str,
                       help='the path to the input qrcode folder')
    parser.add_argument('--output',
                       metavar='outputpath',default="./output_rgbd",
                       type=str,
                       help='output rgbd folder path')
    args = parser.parse_args()


    folders=os.listdir(args.input)
    unique_qr_codes=[os.path.join(args.input,x) for x in folders]
    
    ## make dirs for saving output files
    def output_paths(args_output=args.output):
        seg_output=os.path.join(args_output,"SEG")
        if not os.path.exists(seg_output):
            os.makedirs(seg_output)
        rgbd_output=os.path.join(args_output,"RGBD")
        if not os.path.exists(rgbd_output):
            os.makedirs(rgbd_output)
        return seg_output,rgbd_output

    utils.multiprocess(unique_qr_codes,
        process_method = update_qrs, 
        process_individial_entries  = False, 
        number_of_workers           = 4,
        pass_process_index          = True, 
        progressbar                 = True, 
        disable_gpu                 =True)
    