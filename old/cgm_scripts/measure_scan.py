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

# python3 measure_scan.py /localssd/20190724_Standardization_AAH/RJ_BMZ_TEST_023/measure/1564044745615

import sys
sys.path.append("..")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from absl import logging
logging._warn_preinit_stderr = 0

import json
import glob
from cgmcore import modelutils, utils
from cgm_database import dbutils
import h5py
import numpy as np
from bunch import Bunch
import random

# Exit if not properly called.
#if len(sys.argv) != 4:
#    print("Please provide the path to a scan, dbconnection file and destination_folder")
#    exit(1)

def store_results(scan_path, db_connection_file, destination_folder):

#db_connection_file = str(sys.argv[2])
#destination_folder = str(sys.argv[3])

# Get the path to the scan.
#scan_path = sys.argv[1]
    scan_path_split = scan_path.split("/")
    scan_qrcode = scan_path_split[-3]
    scan_timestamp = scan_path_split[-1]

    # Get the paths to the artifacts.
    glob_search_path = os.path.join(scan_path, "pc", "*.pcd")
    pcd_paths = glob.glob(glob_search_path)
    if len(pcd_paths) == 0:
        print("No artifacts found. Aborting...")
        exit(1)

    # Prepare results dictionary.
    results = Bunch()
    results.scan = Bunch()
    results.scan.qrcode = scan_qrcode
    results.scan.timestamp = scan_timestamp
    results.model_results = []

    main_connector = dbutils.connect_to_main_database(db_connection_file)

    # Select models from model table where active=True in json_metadata
    select_models = "SELECT * FROM model WHERE (json_metadata->>'active')::BOOLEAN IS true;"
    models = main_connector.execute(select_models, fetch_all=True)

    # Go through the models from the models-file.
    #with open("/home/mmatiaschek/whhdata/models.json") as json_file:
    for model in models:
        model_name = model[0]

        # Locate the weights of the model.
        weights_search_path = os.path.join("/home/smahale/whhdata/models", model_name, "*")
        weights_paths = [x for x in glob.glob(weights_search_path) if "-weights" in x]
        if len(weights_paths) == 0:
            continue
        weights_path = weights_paths[0]
        entry = model[3]

        # Get the model parameters.
        input_shape = entry["input_shape"]
        output_size = entry["output_size"]
        hidden_sizes = entry["hidden_sizes"]
        #hidden_sizes = [512, 256, 128]
        subsampling_method = entry["subsampling_method"]

        # Load the model.
        #print(weights_path, input_shape, output_size, hidden_sizes)
        try:
            model = modelutils.load_pointnet(weights_path, input_shape, output_size, hidden_sizes)
        except:
            print("Failed!", weights_path)
            continue
        #print("Worked!", weights_path)

        # Prepare the pointclouds.
        pointclouds = []
        for pcd_path in pcd_paths:
            pointcloud = utils.load_pcd_as_ndarray(pcd_path)
            pointcloud = utils.subsample_pointcloud(
                pointcloud,
                target_size=input_shape[0],
                subsampling_method="sequential_skip")
            pointclouds.append(pointcloud)
        pointclouds = np.array(pointclouds)

        # Predict.
        predictions = model.predict(pointclouds)

        # Prepare model result.
        model_result = Bunch()
        model_result.model_name = model_name

        # Store measure result.
        model_result.measure_result = Bunch()
        model_result.measure_result.mean = str(np.mean(predictions))
        model_result.measure_result.min = str(np.min(predictions))
        model_result.measure_result.max = str(np.max(predictions))
        model_result.measure_result.std = str(np.std(predictions))

        # Store artifact results.
        model_result.artifact_results = []
        for pcd_path, prediction in zip(pcd_paths, predictions):
            artifact_result = Bunch()
            #artifact_result.path = pcd_path
            artifact_result.path = '/'.join(pcd_path.split('/')[4:])
            artifact_result.prediction = str(prediction[0])
            model_result.artifact_results.append(artifact_result)

        results.model_results.append(model_result)

    results_json_string = json.dumps(results)
    #print(results_json_string)

    results_json_object = json.loads(results_json_string)

    filename = "{0}/{1}-{2}-{3}-{4}.json".format(destination_folder, pcd_paths[0].split('/')[3], scan_qrcode, scan_timestamp, random.randint(10000, 99999))

    # Add the results to a json file in destination_folder
    with open(filename, 'w') as json_file:
        json.dump(results_json_object, json_file, indent=2)
