import sys
sys.path.insert(0, "..")

#from cgm_fusion import calibration
from cgm_fusion import fusion
from cgm_fusion import utility

import numpy


import sys
sys.path.insert(0, "..")
import dbutils 
from cgmcore import utils
import matplotlib.pyplot as plt
import pandas as pd
import glob2
import os
import config
from tqdm import tqdm
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.display import display, clear_output
import itertools
from tqdm import tqdm
import logging







def update_depth(pcd_paths, process_index):
    # initialize the rrogress bar with the maxium number of unique qr codes
    #bar = progressbar.ProgressBar(max_value=len(unique_qr_codes))
    # qr_counter = 0



    # print(pcd_paths)

    # for path in pcd_paths:
    for path in tqdm(pcd_paths,position=process_index):

        pcd_path      = path[0]
        pcd_input     = pcd_path
        pcd_path      = pcd_path.replace('localssd', 'whhdata')
        pcd_name      = os.path.basename(pcd_path)
        folder        = os.path.dirname(pcd_path)
        parent_folder = os.path.dirname(folder)

        depth_folder = parent_folder + '/depth'

        
        # print("depth folder: ")
        # print (depth_folder)
        # print("parent folder: ")
        # print(parent_folder)

        # output to 2nd ssd to speed up things
        depth_folder = depth_folder.replace('whhdata', 'localssd2')
        # check if depth folder exists
        if not os.path.exists(depth_folder):
            os.mkdir(depth_folder)
        
        np_path  = depth_folder + "/" + pcd_name     # *.npy
        np_path  = np_path.replace('.ply', '.npy')
        # png_path = depth_folder + "/" + pcd_name     # *.png
        # png_path = png_path.replace('ply', 'png')


        # png_path = png_path.replace('.png', '_all_channels.png')


        # print(np_path)

        # check if the png already exists and in this case continue with the next file
        # if os.path.exists(np_path): 
        #     # print("skipping this path" + str(png_path))
        #     continue
        
        # fusion.get_depth_image_from_point_cloud(calibration_file="dummy", pcd_file="/tmp/cloud_debug.ply", output_file="dummy")
        # utility.get_depth_channel(ply_path=pcd_input, output_path_np=np_path, output_path_png=png_path)
        utility.get_all_channel(ply_path=pcd_input, output_path_np = np_path)


def main():
    # Getting the models.
    db_connector = dbutils.connect_to_main_database()

    # get the number of rgb artifacts
    select_sql_statement = "SELECT path FROM artifact WHERE type='pcrgb';"
    pcd_paths = db_connector.execute(select_sql_statement, fetch_all=True) #[0][0]



    print ('Available fused data: ' + str(len(pcd_paths)))

    # # todo: remove the (1) or (2) backup ?
    # unique_qr_codes = [x[0] for x in unique_qr_codes]

    # initialze log file
    logging.basicConfig(filename='/tmp/command_update_depth.log',level=logging.DEBUG, format='%(asctime)s %(message)s')

    # Run this in multiprocess mode.
    utils.multiprocess(pcd_paths, 
        process_method              = update_depth, 
        process_individial_entries  = False, 
        number_of_workers           = 8,
        pass_process_index          = True, 
        progressbar                 = False, 
        disable_gpu                 = True)
    
    print("*** Done ***.")





if __name__ == "__main__":
    main()

