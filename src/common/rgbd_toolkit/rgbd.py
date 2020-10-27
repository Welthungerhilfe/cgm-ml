import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import azureml.core
from azureml.core.dataset import Dataset
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from glob import glob
from pathlib import Path
# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)
import sys
sys.path.append('../cgm-ml')
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import pickle
from numpy import size
import pandas as pd
from tqdm import tqdm
from segmentation import DeepLabModel,load_model,apply_segmentation
from get_timestamps import get_timestamps_from_rgb,get_timestamps_from_pcd
import cgm_fusion.utility as utility
import cgm_fusion.calibration 
from cgm_fusion.fusion import apply_fusion,fuse_rgbd 
# import core packages from cgm
from cgmcore.utils import load_pcd_as_ndarray
from cgmcore import  utils
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import warnings
warnings.filterwarnings("ignore")
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



def process_pcd(norm_pcd_time,process_index=0):
    i = 0

    for pcd in tqdm(norm_pcd_time,position=process_index):

            #find the closest rgb file for the pcd
            nn = find_closest(norm_rgb_time, pcd)
            logging.info("timestamp of pcd: " + "{0:.2f}".format(round(pcd,2))+ " with index " + str(i)  + " path: " + str(pcd_path[i]))
            logging.info("timestamp of jpg: " + "{0:.2f}".format(round(norm_rgb_time[nn],2)) + " with index " + str(nn) + " path: " + str(rgb_path[nn]))
            
            #get the original file path 
            path, filename = os.path.split(str(pcd_path[i]))

            #located the nearest jpg file of that pcd
            pcd_file = pcd_path[i]
            jpg_file = rgb_path[nn]
            
            #rotating and aligning rgb image
            img_name=jpg_file.split("/")[-1]
            img=Image.open(jpg_file)
    
            
            if '_100_' in img_name or '_101_' in img_name or '_102_' in img_name:
                image = img.rotate(angle=270)
                
            elif '_200_' in img_name or '_201_' in img_name or '_202_' in img_name:
                image = img.rotate(angle=90)
                
            
            qr_folder=str(Path(qr)).split("/")[-1]

            ##TODO:uncomment for segmentation fusion

            # check if a segmentation for the found jpg exists
            # seg_path_ = jpg_file.replace('.jpg', '_SEG.png')
            # seg_path_,seg_file=os.path.split(seg_path_)  
            # seg_folder=os.path.join(seg_folder_,qr_folder)

            # if not os.path.exists(seg_folder):
            #     os.mkdir(seg_folder)
            # seg_path=os.path.join(seg_folder,seg_file)
            
            # if not( os.path.exists(seg_path) ):
            
            #     logging.debug('applying segmentation')
            #     seg_path = apply_segmentation(image,seg_path,model)
            #     # check if the path now exists
            # if not( os.path.exists(seg_path) ):
            #         logging.error('Segmented file does not exist: ' + seg_path)

            i = i+1
            
            calibration_file="./calibration.xml"
           

            # now save the new data to the folder
            fused_folder, pc_filename = os.path.split(str(pcd_file))
            pcd_path_old = pcd_file

            #finding if its standing or lying artifact
            scan_type=pcd_file.split("/")[-1].split("_")[3]
            
            
            
            # replace the pcd and the pc_ in the path for fused data
            pc_filename = pcd_path_old.replace(".pcd", ".ply")
            pc_filename = pc_filename.replace("pc_",   "pcrgb_")
            
            #make a folder for scan type
            qr_rgbd=os.path.join(rgbd_folder_,qr_folder)
            qr_rgbd_scan=os.path.join(qr_rgbd,str(scan_type))
            
            if args.mounted:
            #get height and weight label for the corresponding artifact
                height=int(artifacts_file.loc[np.where(artifacts_file["qrcode"]==qr_folder)].iloc[0].loc["height"])
                weight=int(artifacts_file.loc[np.where(artifacts_file["qrcode"]==qr_folder)].iloc[0].loc["weight"])
            
            #make the output rgbd filename
            rgbd_filename = pc_filename.replace(fused_folder.split("measure")[0], qr_rgbd_scan+"/")
            
    
            # write the data to the new storage
            rgbd_filename=rgbd_filename.replace(".ply",".pkl")
            rgbd_filename=rgbd_filename.replace("/pc/","")  
            
            #check if output rgbd folder exists
            rgbd_folder=os.path.dirname(rgbd_filename)
            if not(os.path.isfile(rgbd_folder)): 
                logging.info("Folder does not exist for " + str(rgbd_filename))
                os.makedirs(rgbd_folder, exist_ok=True)
                logging.info("Created folder " + str(rgbd_folder))


            logging.info("Going to writing new fused data to: " + rgbd_filename)

            #checking if file already exists and saving the rgbd file with labels if its mounted data
            if not os.path.exists(rgbd_filename):
                try: 
                    rgbdseg_arr = fuse_rgbd(calibration_file,pcd_file, image)#, seg_path)
                    if args.mounted:
                        labels=np.array([height,weight])
                        data=(rgbdseg_arr,labels)
                        pickle.dump(data, open(rgbd_filename, "wb"))
                    else:
                    #saving as a png file if not mounted data
                        rgbd_filename=rgbd_filename.replace(".pkl",".png")
                        fig = plt.figure()
                        plt.imsave(rgbd_filename,rgbdseg_arr)
                        #np.save(rgbd_filename,rgbdseg_arr)

                    logging.info("successfully wrote new data to" + rgbd_filename)                              
                except Exception as e: 
                    logging.error("Something went wrong.Skipping this file" + pc_filename)
                    
                    logging.error(str(e))
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
    parser.add_argument("--w",metavar="workers",type=int,default=None,help="no. of cpu workers you want to process with")
    parser.add_argument("--mounted",action='store_true',help="if you are processing on mounted data of a datastore")
    parser.add_argument("--segmented",action='store_true',help="if you want fused rgbd with segmentation")
    args = parser.parse_args()


    start=datetime.datetime.now()

    #reading artifacts.csv for mounted qrcode paths
    if args.mounted:
        artifacts=os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'data_utils/dataset_EDA/50k_pcd/artifacts.csv')
        mnt=args.input+"/qrcode/"
        artifacts_file=pd.read_csv(artifacts)
        qrcode=artifacts_file["qrcode"]
        paths=artifacts_file["storage_path"]
        unique_qr_codes=[os.path.join(mnt,line) for line in qrcode]
        
    else:
        folders=os.listdir(args.input)
        unique_qr_codes=[os.path.join(args.input,x) for x in folders]
    
    ##TODO: streamline validation of input directory
    if not os.path.exists(unique_qr_codes[0]):
        print("Error:invalid input paths..exiting")
        exit()
        
    #loading DeepLab model
    if args.segmented:
        model=load_model()

    
    #making output dir for storing rgbd files
    rgbd_folder_=args.output
    if not os.path.exists(rgbd_folder_):
        os.mkdir(rgbd_folder_)


    #initialize empty lists for rgb and pcd paths of a given qr code
    rgb_paths=[]
    pcd_paths=[]

    #read jpg and pcd files of each qr code in the input directory.
    print("Processing..")
    start=datetime.datetime.now()
    for qr in set(unique_qr_codes):
        logging.info("reading qr code"+ str(qr))      
        for dirname,dirs,qr_paths in os.walk(Path(qr)):
                for file in qr_paths :
                        dir_path=os.path.join(dirname,file)
                        if file.endswith(".jpg"):
                            rgb_paths.append(dir_path)
                        if file.endswith(".pcd"):
                            pcd_paths.append(dir_path)
        
            
    #getting the timestamps of rgb and pcd paths
        [norm_rgb_time, rgb_path] = get_timestamps_from_rgb(rgb_paths)
        [norm_pcd_time, pcd_path] = get_timestamps_from_pcd(pcd_paths)

        if ( size(norm_rgb_time) == 0 ):
            logging.error("wrong size of jpg")
            logging.error("size rgb: " + str(size(norm_rgb_time)))
            continue

        if ( size(norm_pcd_time) == 0 ): 
            logging.error("wrong size of pcd")    
            logging.error("size pcd: " + str(size(norm_pcd_time)))
            continue


    #processing every pcd file with its nearest rgb using multiprocessing workers
        utils.multiprocess(norm_pcd_time,
        process_method = process_pcd, 
        process_individial_entries  = False, 
        number_of_workers           = args.w,
        pass_process_index          = True, 
        progressbar                 = True, 
        disable_gpu                 =True)
        
        
    end = datetime.datetime.now()
    diff = end - start
    print("***Done***")
    print("total time took is {}".format(diff))
    logging.info("total time took is"+ str(diff))


    