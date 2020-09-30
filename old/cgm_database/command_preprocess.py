#
# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Tristan Behrens <tristan@ai-guru.de> for Welthungerhilfe
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

import warnings
warnings.filterwarnings("ignore")
import dbutils
import os
import sys
sys.path.insert(0, "..")
from cgmcore import utils
import numpy as np
import cv2
import datetime
import pickle
import config


def execute_command_preprocess(preprocess_pcds=True, preprocess_ply=False, preprocess_jpgs=False, preprocess_npy=False, preprocess_png=False, path_suffix=""):
    print("Preprocessing data-set...")
    
    print("Using '{}'".format(config.preprocessed_root_path))
    if os.path.exists(config.preprocessed_root_path) == False:
        print("Folder does not exists. Creating...")
        os.mkdir(config.preprocessed_root_path)
    

    # Create the base-folder.
    if path_suffix != "":
        path_suffix = "-" + path_suffix
    datetime_path = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_path = os.path.join(config.preprocessed_root_path, datetime_path + path_suffix)
    os.mkdir(base_path)


    if preprocess_pcds == True:
        os.mkdir(os.path.join(base_path, "pcd"))
    if preprocess_ply == True:
        os.mkdir(os.path.join(base_path, "ply"))
    if preprocess_jpgs == True:
        os.mkdir(os.path.join(base_path, "jpg"))
    if preprocess_npy == True:
        os.mkdir(os.path.join(base_path, "npy"))


    print("Writing preprocessed data to {}...".format(base_path))
    

    # Process the filtered PCDs.
    if preprocess_pcds == True:
        
        # Get entries.
        sql_statement = """
            SELECT artifact_path, qr_code, height, weight 
            FROM artifacts_with_targets 
            WHERE type='pcd'
            AND status='standing'
            ;
            """
        main_connector = dbutils.connect_to_main_database()
        entries = main_connector.execute(sql_statement, fetch_all=True)
        print("Found {} PCDs. Processing...".format(len(entries)))
        
        # Method for processing a single entry.
        def process_pcd_entry(entry):
            path, qr_code, height, weight = entry
            if os.path.exists(path) == False:
                print("\n", "File {} does not exist!".format(path), "\n")
                return
            try:
                pointcloud = utils.load_pcd_as_ndarray(path)
                targets = np.array([height, weight])
                pickle_filename = os.path.basename(path).replace(".pcd", ".p")
                qrcode_path = os.path.join(base_path, "pcd", qr_code)
                if os.path.exists(qrcode_path) == False:
                    os.mkdir(qrcode_path)
                pickle_output_path = os.path.join(qrcode_path, pickle_filename)
                pickle.dump((pointcloud, targets), open(pickle_output_path, "wb"))
            except:
                pass
        
        # Start multiprocessing.
        utils.multiprocess(entries, process_pcd_entry)


    if preprocess_ply == True:
        
        # Get entries.
        sql_statement = """
            SELECT artifact_path, qr_code, height, weight 
            FROM artifacts_with_targets 
            WHERE type='pcrgb'
            AND status='standing'
            ;
            """
        main_connector = dbutils.connect_to_main_database()
        entries = main_connector.execute(sql_statement, fetch_all=True)
        print("Found {} Plys. Processing...".format(len(entries)))
        
        # Method for processing a single entry.
        def process_ply_entry(entry):
            path, qr_code, height, weight = entry

            print(path)

            if os.path.exists(path) == False:
                print("\n", "File {} does not exist!".format(path), "\n")
                return

                
            try:
                pointcloud = utils.load_pcd_as_ndarray(path)
                targets = np.array([height, weight])
                pickle_filename = os.path.basename(path).replace(".ply", ".p")
                qrcode_path = os.path.join(base_path, "ply", qr_code)

#                print('qr code path: ' + qrcode_path)

               # if not os.path.exists(qrcode_path):
                os.mkdir(qrcode_path)
                print('creating path')


                pickle_output_path = os.path.join(qrcode_path, pickle_filename)
                pickle.dump((pointcloud, targets), open(pickle_output_path, "wb"))
            except BaseException as e:
                print(e) 
                pass
        
        # Start multiprocessing.
        utils.multiprocess(entries, process_ply_entry)



    # Process the filtered JPGs.
    if preprocess_jpgs == True:
        assert False
        entries = filterjpgs()["results"]
        print("Found {} JPGs. Processing...".format(len(entries)))
        bar = progressbar.ProgressBar(max_value=len(entries))
        
        # Method for processing a single entry.
        def process_jpg_entry(entry):
            path = entry["path"]
            if os.path.exists(path) == False:
                print("\n", "File {} does not exist!".format(path), "\n")
                return
            image = cv2.imread(path)
            targets = np.array([entry["height_cms"], entry["weight_kgs"]])
            qrcode = entry["qrcode"]
            pickle_filename = os.path.basename(entry["path"]).replace(".jpg", ".p")
            qrcode_path = os.path.join(base_path, "jpg", qrcode)
            if os.path.exists(qrcode_path) == False:
                os.mkdir(qrcode_path)
            pickle_output_path = os.path.join(qrcode_path, pickle_filename)
            pickle.dump((image, targets), open(pickle_output_path, "wb"))
        
        # Start multiprocessing.
        utils.multiprocess(entries, process_pcd_entry)


    # Process depth images numpy array
    if preprocess_npy == True:
        
        # Get entries.
        sql_statement = """
            SELECT artifact_path, qr_code, height, weight 
            FROM artifacts_with_targets 
            WHERE type='depth_npy'
            AND status='standing'
            ;
            """
        main_connector = dbutils.connect_to_main_database()
        entries = main_connector.execute(sql_statement, fetch_all=True)
        print("Found {} depth_npy. Processing...".format(len(entries)))
        
        # Method for processing a single entry.
        def process_pcd_entry(entry):
            path, qr_code, height, weight = entry
            if os.path.exists(path) == False:
                print("\n", "File {} does not exist!".format(path), "\n")
                return
            try:
                print(path)
                np_array = np.load(path)
                print("opened file")
                targets = np.array([height, weight])
                pickle_filename = os.path.basename(path).replace(".npy", ".p")
                qrcode_path = os.path.join(base_path, "npy", qr_code)
                if os.path.exists(qrcode_path) == False:
                    os.mkdir(qrcode_path)
                pickle_output_path = os.path.join(qrcode_path, pickle_filename)
                pickle.dump((np_array, targets), open(pickle_output_path, "wb"))
            except:
                print("exception fails")
                pass
        
        # Start multiprocessing.
        utils.multiprocess(entries, process_pcd_entry)

    # print(preprocess_png)
    # preprocess_png = True
    # Process the filtered depth pngs.
    if preprocess_png == True:
        

        # Get entries.
        sql_statement = """
            SELECT artifact_path, qr_code, height, weight 
            FROM artifacts_with_targets 
            WHERE type='depth_png'
            AND status='standing'
            ;
            """
        main_connector = dbutils.connect_to_main_database()
        entries = main_connector.execute(sql_statement, fetch_all=True)
        print("Found {} depth_npy. Processing...".format(len(entries)))
        
        # Method for processing a single entry.
        def process_pcd_entry(entry):
            path, qr_code, height, weight = entry
            if os.path.exists(path) == False:
                print("\n", "File {} does not exist!".format(path), "\n")
                return
            try:
                print(path)
                np_array = cv2.imread(path,  cv2.CV_8U)
                print(type(np_array))
                print("opened file")
                print(np_array.shape)
                targets = np.array([height, weight])
                pickle_filename = os.path.basename(path).replace(".png", ".p")
                qrcode_path = os.path.join(base_path, "png", qr_code)
                print(qrcode_path)
                # if os.path.exists(qrcode_path) == False:
                try:
                    os.makedirs(qrcode_path, exist_ok=True)
                except: 
                    print("can not create folder")
                    pass
                    # print("path created")
                    # os.mkdir(qrcode_path)
                print("after path created")

                print("in advance to pickle")
                pickle_output_path = os.path.join(qrcode_path, pickle_filename)
                print("created pickle")
                pickle.dump((np_array, targets), open(pickle_output_path, "wb"))
            except:
                print("Failed to import image")
                pass
        
        # Start multiprocessing.
        utils.multiprocess(entries, process_pcd_entry)
        
# TODO remove this soon        
def filterpcds(
    number_of_points_threshold=10000, 
    confidence_avg_threshold=0.75,
    remove_unreasonable=True,
    remove_errors=True, 
    remove_rejects=True, 
    sort_key=None, 
    sort_reverse=False):
    
    print("Filtering DB...")
    
    sql_statement = ""
    # Get all pointclouds.
    sql_statement += "SELECT * FROM artifacts_with_targets WHERE type='pcd'"
    
    # TODO sql_statement += " WHERE number_of_points > {}".format(number_of_points_threshold) 
    
    # Remove pointclouds that have a confidence that is too low.
    # TODO sql_statement += " AND confidence_avg > {}".format(confidence_avg_threshold)
    
    # Ignore measurements that are not plausible.
    if remove_unreasonable == True:
        sql_statement += " AND height >= 60"
        sql_statement += " AND height <= 120"
        sql_statement += " AND weight >= 2"
        sql_statement += " AND weight <= 20"
    
    # Execute statement.
    results = main_connector.execute(sql_statement, fetch_all=True)
    columns = []
    columns.extend(main_connector.get_columns("artifacts_with_targets"))
    results = [dict(list(zip(columns, result))) for result in results]
    return { "results" : results }

        
def filterjpgs(
    blur_variance_threshold=100.0,
    remove_errors=True, 
    remove_rejects=True, 
    sort_key=None, 
    sort_reverse=False):
    
    print("Filtering DB...")
    
    sql_statement = ""
    sql_statement += "SELECT * FROM {}".format(IMAGES_TABLE)
    sql_statement += " INNER JOIN measurements ON {}.measurement_id=measurements.id".format(IMAGES_TABLE)
    sql_statement += " WHERE blur_variance > {}".format(blur_variance_threshold) 
    sql_statement += " AND measurements.type=\'manual\'"
    if remove_errors == True:
        sql_statement += " AND had_error = false" 
    if remove_rejects == True:
        sql_statement += " AND rejected_by_expert = false" 
    if sort_key != None:
        sql_statement += " ORDER BY {}".format(sort_key) 
        if sort_reverse == False:
            sql_statement += " ASC" 
        else:
            sql_statement += " DESC"

    results = main_connector.execute(sql_statement, fetch_all=True)
    columns = []
    columns.extend(main_connector.get_columns(IMAGES_TABLE))
    columns.extend(main_connector.get_columns(MEASUREMENTS_TABLE))
    results = [dict(list(zip(columns, result))) for result in results]
    return { "results" : results }
    

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        raise Exception("ERROR! Must specify what to update. [images|pointclouds|fusion|depth|all]")

    # Parse command line arguments.
    preprocess_pcds = False
    preprocess_jpgs = False
    preprocess_ply  = False
    preprocess_npy  = False
    preprocess_png  = False

    if sys.argv[1] == "images":
        print("Updating images only...")
        preprocess_jpgs = True
    elif sys.argv[1] == "pointclouds":
        print("Updating pointclouds only...")
        preprocess_pcds = True
    elif sys.argv[1] == "fusion":
        print("Updating fusion only...")
        preprocess_ply = True
    # elif sys.argv[1] == "depth":
    #     print("Updating depth only...")
        # preprocess_npy = True
    elif sys.argv[1] == "depth":
        print("Updating depth only...")
        preprocess_npy = True
    elif sys.argv[1] == "all":
        print("Updating all...")
        preprocess_jpgs = True
        preprocess_pcds = True
        preprocess_ply  = True
        preprocess_npy  = True
        preprocess_png  = True
    
    path_suffix = ""
    if len(sys.argv) > 2:
        path_suffix = sys.argv[2]
    
    execute_command_preprocess(preprocess_pcds, preprocess_ply, preprocess_jpgs, preprocess_npy, preprocess_png, path_suffix)
