import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import azureml.core
from azureml.core.dataset import Dataset
import json
import os
from glob import glob
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

# import command_update_segmentation

# import core packages from cgm
from cgmcore.utils import load_pcd_as_ndarray
from cgmcore import  utils


import datetime
import matplotlib.pyplot as plt


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

def update_qrs(unique_qr_codes, rgb_paths,pcd_paths,mount_path,output):
    
    qr_counter = 0

    
    
    # load model for segmentation
    
    modelType = "./xception_model"
    MODEL     = DeepLabModel(modelType)
    logging.info('model loaded successfully : ' + modelType)
    
    for qr in unique_qr_codes:

        qr_counter = qr_counter + 1
        if qr == "{qrcode}":
            continue
        if qr == "data":
            continue


        [norm_rgb_time, rgb_path] = get_timestamps_from_rgb(qr,rgb_paths)
        [norm_pcd_time, pcd_path] = get_timestamps_from_pcd(qr,pcd_paths)
        
        # check if a qr code has rgb and pcd, otherwise the previous function returned -1

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
            
            nn = find_closest(norm_rgb_time, pcd)
            logging.info("timestamp of rgb: " + "{0:.2f}".format(round(pcd,2))               + " with index " + str(i)) # + " path: " + str(pcd_path[i]))
            logging.info("timestamp of jpg: " + "{0:.2f}".format(round(norm_rgb_time[nn],2)) + " with index " + str(nn))# + " path: " + str(rgb_path[nn]))
            
            # get the original file path 
            path, filename = os.path.split(str(pcd_path[i]))

            pcd_file = pcd_path[i]
            #pcd_file = pcd_file[0]
            jpg_file = rgb_path[nn]
            
            
            # check if a segmentation for the found jpg exists
            seg_path = jpg_file.replace('.jpg', '_SEG.png')
            if not( os.path.exists(seg_path) ):
                
                logging.debug('applying segmentation')
                seg_path = apply_segmentation(jpg_file, MODEL)
                #check if the path now exists
            if not( os.path.exists(seg_path) ):
                    logging.error('Segmented file does not exist: ' + seg_path)

            i = i+1

            calibration_file="./calibration.xml"
            
            
            # the point cloud and rgbd data is fused and additionally the cloud is saved as ply and .npy respectively
            #fused_cloud = apply_fusion(calibration_file,pcd_file, jpg_file, seg_path)
            #rgbdseg_arr = fuse_rgbd(calibration_file,pcd_file, jpg_file, seg_path)
            
            try: 
                
                #fused_cloud = apply_fusion(calibration_file,pcd_file, jpg_file, seg_path)
                rgbdseg_arr = fuse_rgbd(calibration_file,pcd_file, jpg_file, seg_path)
            
                                           
            except Exception as e: 
                logging.error("Something went wrong. ")
                
                logging.error(str(e))
                print("error",e)
                continue

            # now save the new data to the folder
            fused_folder, pc_filename = os.path.split(str(pcd_file))
            pcd_path_old = pcd_file
            
            # replace the pcd and the pc_ in the path for fused data
            pc_filename = pcd_path_old.replace(".pcd", ".ply")
            pc_filename = pc_filename.replace("pc_",   "pcrgb_")

            
            

            # write the data to the new storage
            pc_filename = pc_filename.replace(mount_path, output)
            pc_png=pc_filename.replace(".ply",".npy")
            rgb_filename=pc_png.replace("/pc/","/RGBD/")
            
            


            #check if folder exists
            # pc_folder = os.path.dirname(pc_filename)
            # if not(os.path.isfile(pc_folder)): 
            #     logging.info("Folder does not exist for " + str(pc_filename))
            #     os.makedirs(pc_folder, exist_ok=True)
            #     logging.info("Created folder " + str(pc_folder))

            rgb_folder=os.path.dirname(rgb_filename)
            if not(os.path.isfile(rgb_folder)): 
                logging.info("Folder does not exist for " + str(rgb_filename))
                os.makedirs(rgb_folder, exist_ok=True)
                logging.info("Created folder " + str(rgb_folder))


            logging.info("Going to writing new fused data to: " + pc_filename)

            
             
            try: 
                #fused_cloud.to_file(pc_filename)
                np.save(rgb_filename,rgbdseg_arr)
                # fig=plt.figure() 
                # #plt.imshow(rgbdseg_arr)
                # plt.imsave(rgb_filename,rgbdseg_arr) 
                    
            except AttributeError :
                logging.error("An error occured -- skipping this file to save ") 
                continue

def main(args):
    
    mount_path=args.input
    rgb_paths=[]
    pcd_paths=[]
    for qr in os.listdir(mount_path):
    
        for dirname,dirs,files in os.walk(os.path.join(mount_path,qr)):
      
            for file in files:
                dir_path=os.path.join(dirname,file)
                if file.endswith(".jpg"):
                    rgb_paths.append(dir_path)
                if file.endswith(".pcd"):
                    pcd_paths.append(dir_path)



    unique_qr_codes=os.listdir(mount_path)
    update_qrs(unique_qr_codes,rgb_paths,pcd_paths,mount_path,output=args.output)

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


    main(args)