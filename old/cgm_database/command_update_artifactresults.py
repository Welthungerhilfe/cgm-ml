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
import sys
sys.path.insert(0, "..")
import os
import dbutils
import math
import numpy as np
from cgmcore import modelutils, utils
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import progressbar
import psycopg2
import glob
import pickle
import cv2
import posenet
import tensorflow as tf
import time
from tqdm import tqdm


def main():

    # Check the arguments.
    if len(sys.argv) != 2:
        raise Exception("ERROR! Must provide model filename.")
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        raise Exception("ERROR! \"{}\" does not exist.".format(model_path))
        
        
    
    
    # Get the training QR-codes.
    #search_path = os.path.join(os.path.dirname(model_path), "*.p")
    #paths = glob.glob(search_path)
    #details_path = [path for path in paths if "details" in path][0]
    #details = pickle.load(open(details_path, "rb"))
    #qrcodes_train = details["qrcodes_train"]
    #qrcodes_validate = details["qrcodes_validate"]
    #print("QR codes train:", len(qrcodes_train), "QR codes validate:", len(qrcodes_validate))

    # Query the database for artifacts.
    print("Getting all artifacts...")
    db_connector = dbutils.connect_to_main_database()
    sql_statement = ""
    sql_statement += "SELECT id, path FROM artifact"
    sql_statement += " WHERE type='pcd'"
    sql_statement += ";"
    artifacts = db_connector.execute(sql_statement, fetch_all=True)
    print("Found {} artifacts.".format(len(artifacts)))
    
    # Method for processing a set of artifacts.
    # Note: This method will run in its own process.
    def process_artifacts(artifacts, process_index):
        
         # Create database connection.
        db_connector = dbutils.connect_to_main_database()
    
        # Load the model first.
        model_weights_path = [x for x in glob.glob((os.path.join(model_path, "*"))) if x.endswith("-model-weights.h5")][0]
        model_details_path = [x for x in glob.glob((os.path.join(model_path, "*"))) if x.endswith("-details.p")][0] 
        model_name = model_path.split("/")[-1]
        model_details = pickle.load(open(model_details_path, "rb"))
        pointcloud_target_size = model_details["dataset_parameters"]["pointcloud_target_size"]
        pointcloud_subsampling_method = model_details["dataset_parameters"]["pointcloud_subsampling_method"]
        target_key = model_details["dataset_parameters"]["output_targets"][0]
        model = load_model(model_weights_path, pointcloud_target_size)
        
        # Evaluate and create SQL-statements.
        for artifact_index, artifact in enumerate(tqdm(artifacts, position=process_index)):

            # Unpack fields.
            artifact_id, pcd_path = artifact
            
            # Check if there is already an entry.
            select_sql_statement = ""
            select_sql_statement += "SELECT COUNT(*) FROM artifact_result"
            select_sql_statement += " WHERE artifact_id='{}'".format(artifact_id)
            select_sql_statement += " AND model_name='{}'".format(model_name)
            select_sql_statement += " AND target_key='{}'".format(target_key)
            results = db_connector.execute(select_sql_statement, fetch_one=True)[0]

            # There is an entry. Skip
            if results != 0:
                continue
            
            # Execute SQL statement.
            try:
                # Load the artifact and evaluate.
                pcd_path = pcd_path.replace("/whhdata/qrcode", "/localssd/qrcode")
                pcd_array = utils.load_pcd_as_ndarray(pcd_path)
                pcd_array = utils.subsample_pointcloud(pcd_array, pointcloud_target_size, subsampling_method=pointcloud_subsampling_method)
                
                value = model.predict(np.expand_dims(pcd_array, axis=0), verbose=0)[0][0]
                
                # Create an SQL statement.
                sql_statement = ""
                sql_statement += "INSERT INTO artifact_result (model_name, target_key, value, artifact_id)"
                sql_statement += " VALUES(\'{}\', \'{}\', \'{}\', \'{}\');".format(model_name, target_key, value, artifact_id)

                # Call database.
                result = db_connector.execute(sql_statement)
            except psycopg2.IntegrityError as e:
                #print("Already in DB. Skipped.", pcd_path)
                pass
            except ValueError as e:
                #print("Skipped.", pcd_path)
                pass

    # Run this in multiprocess mode.
    utils.multiprocess(
        artifacts, 
        process_method=process_artifacts, 
        process_individial_entries=False, 
        progressbar=False,
        pass_process_index=True,
        disable_gpu=True
    )
    print("Done.")
        
        
def load_model(model_weights_path, pointcloud_target_size):

    input_shape = (pointcloud_target_size, 3)
    output_size = 1
    model = modelutils.create_point_net(input_shape, output_size, hidden_sizes = [512, 256, 128])
    model.load_weights(model_weights_path)
    model.compile(
        optimizer="rmsprop",
        loss="mse",
        metrics=["mae"]
    )
    return model 

if __name__ == "__main__":
    main()