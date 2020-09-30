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
import glob2 as glob
import dbutils
import progressbar
import time
import datetime
import cv2
from cgmcore import utils
import hashlib
import config
from tqdm import tqdm
import pickle


extension_to_type = {
    "pcd" : "pcd",
    "jpg" : "rgb",
    "ply" : "pcrgb",
    "png" : "depth_png",
    "npy" : "depth_npy"
}


def execute_command_update_artifacts(update_jpgs=False, update_pcds=False, update_pcrgb=True, update_depth_npy=False, update_depth_png=False):

    # Get all persons.
    print("Finding all persons at '{}'...".format(config.artifacts_path))
    person_search_path = os.path.join(config.artifacts_path, "*")
    person_paths = [path for path in glob.glob(person_search_path) if os.path.isdir(path)]
    
    # TODO speedup... only for development!
    #pickle.dump(person_paths, open("temp.p", "wb"))
    #person_paths = pickle.load(open("temp.p", "rb"))

    #person_paths = person_paths[0:20]
    
    # Deleting the table. Be careful!
    #print("DELETING TABLE!")
    #dbutils.connect_to_main_database().execute("DELETE FROM artifact;")
    
    # Statistics.
    print("Found {} persons.".format(len(person_paths)))
    
    # Decide on the file-extensions.
    file_extensions = []
    if update_jpgs == True:
        file_extensions.append("jpg")
    if update_pcds == True:
        file_extensions.append("pcd")
    if update_pcrgb == True: 
        file_extensions.append("ply")
    if update_depth_npy == True:
        file_extensions.append("npy")
    if update_depth_png == True: 
        file_extensions.append("png")
    
    # This method is executed in multi-processing mode.
    def process_person_paths(person_paths, process_index):
        
        #person_paths = person_paths[0:4] # TODO remove this!
        
        # Go through each person (qr-code).
        for person_path in tqdm(person_paths, position=process_index):
            
            person_path  = person_path.replace('localssd/', 'localssd2/')

            print (person_path)

            # Find all artifacts for that person.
            artifact_paths = []
            for file_extension in file_extensions:
                print(file_extension)
                glob_search_path = os.path.join(person_path, "**/*.{}".format(file_extension))

                
                #print (glob_search_path)
                artifact_paths.extend(glob.glob(glob_search_path))
                # print(artifact_paths)
                
            
            print("Found {} artifacts in {}".format(len(artifact_paths), person_path)) 
        
            # Process those artifacts.
            main_connector = dbutils.connect_to_main_database()
            table = "artifact"
            batch_size = 100
            insert_count = 0
            no_measurements_count = 0
            skip_count = 0
            sql_statement = ""
            last_index = len(artifact_paths) - 1
            for artifact_index, artifact_path in enumerate(artifact_paths):
                
                # Check if there is already an entry in the database.
                basename = os.path.basename(artifact_path)
                sql_statement_select = dbutils.create_select_statement("artifact", ["id"], [basename]) 
                results = main_connector.execute(sql_statement_select, fetch_all=True)

                # No results found. Insert.
                if len(results) == 0:
                    insert_data = {}
                    insert_data["id"] = basename # TODO proper?

                    # Get the default values for the artifact.
                    default_values = get_default_values(artifact_path, table, main_connector)
                    
                    # Check if there is a measure_id.
                    if "measure_id" in default_values.keys():
                        insert_count += 1
                    else:
                        no_measurements_count += 1

                    # Create SQL statement.
                    insert_data.update(default_values)
                    sql_statement_for_artifact = dbutils.create_insert_statement(table, insert_data.keys(), insert_data.values())
                    sql_statement += sql_statement_for_artifact
                        
                # Found a result. Update.
                elif len(results) != 0:
                    skip_count += 1

                # Update database.
                if artifact_index != 0 and ((artifact_index % batch_size) == 0) or artifact_index == last_index:
                    if sql_statement != "":
                        result = main_connector.execute(sql_statement) 
                        sql_statement = ""
                       
        
        # Return statistics.
        return (insert_count, no_measurements_count, skip_count)
        
    
    # Start multiprocessing.
    results = utils.multiprocess(
        person_paths, 
        process_method=process_person_paths, 
        process_individial_entries=False, 
        pass_process_index=True,
        progressbar=False, 
        number_of_workers=6
    )
    
    if results == None:
        print("\n")
        print("No results.")
        return
    
    # Aggregate results
    total_insert_count = 0
    total_no_measurements_count = 0
    total_skip_count = 0
    for (insert_count, no_measurements_count, skip_count) in results: 
        total_insert_count += insert_count
        total_no_measurements_count += no_measurements_count
        total_skip_count += skip_count
    
    print("\n")
    print("Inserted {} new entries.".format(total_insert_count))
    print("No measurements for {} entries.".format(total_no_measurements_count))
    print("Skipped {} entries.".format(total_skip_count))
    
    
def get_artifact_paths(file_extensions):
    config.art
    # Get all persons.
    person_search_path = os.path.join(config.artifacts_path, "*")
    person_paths = [path for path in glob.glob(person_search_path) if os.path.isdir(path)]
    print("Found {} persons.".format(len(person_paths)))
    
    # Method for multiprocessing.
    def process_person_paths(person_paths, process_index):
        artifact_paths_per_process = []
        for person_path in tqdm(person_paths, position=process_index):
            for file_extension in file_extensions:
                glob_search_path = os.path.join(person_path, "**/*.{}".format(file_extension))
                artifact_paths_per_process.extend(glob.glob(glob_search_path))
        return artifact_paths_per_process
    
    # Use multiprocessing.
    artifact_paths = utils.multiprocess(
        person_paths, 
        process_method=process_person_paths, 
        process_individial_entries=False, 
        progressbar=False, 
        pass_process_index=True,
        number_of_workers=None
    )
    return artifact_paths
    
    
def md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
 
    

def get_default_values(file_path, table, db_connector):
     
    # Split and check the path.
    path_split = file_path.split("/")
    #assert path_split[1] == whhdata_path[1:]
    #assert path_split[2] == media_subpath
    
    # Get QR-code and timestamps from path.
    qr_code = path_split[3]
    timestamp = path_split[-1].split("_")[-3]
    tango_timestamp = path_split[-1].split("_")[-1][:-4]
    print(path_split)
    print(qr_code)
    print(timestamp)
    print(tango_timestamp)

    
    # Getting last updated timestamp.
    last_updated, _ = get_last_updated()

    # Get id of measurement.
    threshold = int(60 * 60 * 24 * 1000)
    sql_statement = ""
    sql_statement += "SELECT measure.id"
    sql_statement += " FROM measure"
    sql_statement += " INNER JOIN person ON measure.person_id=person.id"
    sql_statement += " WHERE person.qr_code = '{}'".format(qr_code)
    sql_statement += " AND measure.type = 'manual'"
    sql_statement += " AND ABS(measure.timestamp - {}) < {}".format(timestamp, threshold)
    sql_statement += ";"
    results = db_connector.execute(sql_statement, fetch_all=True)
    
    # Prepare values.
    file_extension = file_path.split(".")[-1]
    values = {}
    values["type"] = extension_to_type[file_extension]
    values["path"] = file_path
    values["hash_value"] = md5(file_path)
    values["file_size"] = os.path.getsize(file_path)
    values["upload_date"] = 0 # TODO make proper
    values["deleted"] = False
    values["qr_code"] = qr_code
    values["create_date"] = timestamp
    values["tango_timestamp"] = tango_timestamp
    values["created_by"] = "UNKNOWN CREATOR" # TODO make proper
    values["status"] = 0 # TODO make proper
    
    # Measurement id not found.
    if len(results) == 0:
        #print("No measure_id found for {}".format(file_path))
        #values["measure_id"] = None
        pass
    
    # Found a measurement id.
    else:
        values["measure_id"] = results[0][0]
    
    return values
    
    
def get_last_updated():
    last_updated = time.time()
    last_updated_readable = datetime.datetime.fromtimestamp(last_updated).strftime('%Y-%m-%d %H:%M:%S')
    return last_updated, last_updated_readable
   

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        raise Exception("ERROR! Must specify what to update. [images|pointclouds|depth|fusion|all]")

    # Parse command line arguments.
    update_jpgs      = False
    update_pcds      = False
    update_pcrgb     = False
    update_depth_png = False
    update_depth_npy = False

    if sys.argv[1] == "images":
        print("Updating images only...")
        update_jpgs = True
    elif sys.argv[1] == "pointclouds":
        print("Updating pointclouds only...")
        update_pcds = True
    elif sys.argv[1] == "fusion": 
        print("Updateing pcrgb only...")
        update_pcrgb = True
    elif sys.argv[1] == "depth":
        print("Updating depth only ...")
        update_depth_npy = True
        # update_depth_png = True
    elif sys.argv[1] == "all":
        print("Updating all artifacts...")
        update_jpgs      = True
        update_pcds      = True
        update_pcrgb     = True
        update_depth_npy = True
        update_depth_png = True
    
    # Run the thing.
    execute_command_update_artifacts(update_jpgs, update_pcds, update_pcrgb, update_depth_npy, update_depth_png)
                        
                        
                        
                        
                        
    