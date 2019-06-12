import warnings
warnings.filterwarnings("ignore")
import dbutils
import os
import sys
sys.path.insert(0, "..")
from cgmcore import utils
import numpy as np
import datetime
import pickle


preprocessed_root_path = "/whhdata/preprocessed"

main_connector = dbutils.connect_to_main_database()
MEASUREMENTS_TABLE = "measurements"
IMAGES_TABLE = "image_data"
POINTCLOUDS_TABLE = "pointcloud_data"


def execute_command_preprocess(preprocess_pcds=True, preprocess_jpgs=False):
    print("Preprocessing data-set...")
    
    # Create the base-folder.
    datetime_path = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_path = os.path.join(preprocessed_root_path, datetime_path)
    os.mkdir(base_path)
    if preprocess_pcds == True:
        os.mkdir(os.path.join(base_path, "pcd"))
    if preprocess_jpgs == True:
        os.mkdir(os.path.join(base_path, "jpg"))
    print("Writing preprocessed data to {}...".format(base_path))
    
    # Process the filtered PCDs.
    if preprocess_pcds == True:
        # Filter parameters.
        number_of_points_threshold=10000
        confidence_avg_threshold=0.75
        remove_unreasonable=True
        remove_errors=True
        remove_rejects=True 
        
        # Save filter parameters.
        filter_parameters_path = os.path.join(base_path, "filter_parameters.txt")
        with open(filter_parameters_path, "w") as filter_parameters_file:
            filter_parameters_file.write("number_of_points_threshold" + "," + str(number_of_points_threshold) + "\n")
            filter_parameters_file.write("confidence_avg_threshold" + "," + str(confidence_avg_threshold) + "\n")
            filter_parameters_file.write("remove_unreasonable" + "," + str(remove_unreasonable) + "\n")
            filter_parameters_file.write("remove_errors" + "," + str(remove_errors) + "\n")
            filter_parameters_file.write("remove_rejects" + "," + str(remove_rejects) + "\n")
        
        # Get filtered entries.
        entries = filterpcds(
            number_of_points_threshold=number_of_points_threshold,
            confidence_avg_threshold=confidence_avg_threshold,
            remove_unreasonable=remove_unreasonable,
            remove_errors=remove_errors,
            remove_rejects=remove_rejects
        )["results"]
        print("Found {} PCDs. Processing...".format(len(entries)))
        
        # Method for processing a single entry.
        def process_pcd_entry(entry):
            path = entry["path"]
            if os.path.exists(path) == False:
                print("\n", "File {} does not exist!".format(path), "\n")
                return
            pointcloud = utils.load_pcd_as_ndarray(path)
            targets = np.array([entry["height_cms"], entry["weight_kgs"]])
            qrcode = entry["qrcode"]
            pickle_filename = os.path.basename(entry["path"]).replace(".pcd", ".p")
            qrcode_path = os.path.join(base_path, "pcd", qrcode)
            if os.path.exists(qrcode_path) == False:
                os.mkdir(qrcode_path)
            pickle_output_path = os.path.join(qrcode_path, pickle_filename)
            pickle.dump((pointcloud, targets), open(pickle_output_path, "wb"))
        
        # Start multiprocessing.
        utils.multiprocess(entries, process_pcd_entry)
    
    # Process the filtered JPGs.
    if preprocess_jpgs == True:
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
    sql_statement += "SELECT * FROM {}".format(POINTCLOUDS_TABLE)
    
    # Join them with measurements.
    sql_statement += " INNER JOIN measurements ON {}.measurement_id=measurements.id".format(POINTCLOUDS_TABLE)
    
    # Remove pointclouds that have to few points.
    sql_statement += " WHERE number_of_points > {}".format(number_of_points_threshold) 
    
    # Only take into account manual measurements.
    sql_statement += " AND measurements.type=\'manual\'"
    
    # Remove pointclouds that have a confidence that is too low.
    sql_statement += " AND confidence_avg > {}".format(confidence_avg_threshold)
    
    # Ignore measurements that are not plausible.
    if remove_unreasonable == True:
        sql_statement += " AND measurements.height_cms >= 60"
        sql_statement += " AND measurements.height_cms <= 120"
        sql_statement += " AND measurements.weight_kgs >= 2"
        sql_statement += " AND measurements.weight_kgs <= 20"
    
    # Remove errors.
    if remove_errors == True:
        sql_statement += " AND had_error = false" 
    
    # Remove rejected samples.
    if remove_rejects == True:
        sql_statement += " AND rejected_by_expert = false" 
    
    # Do some sorting.
    if sort_key != None:
        sql_statement += " ORDER BY {}".format(sort_key) 
        if sort_reverse == False:
            sql_statement += " ASC" 
        else:
            sql_statement += " DESC"
    
    # Execute statement.
    results = main_connector.execute(sql_statement, fetch_all=True)
    columns = []
    columns.extend(main_connector.get_columns(POINTCLOUDS_TABLE))
    columns.extend(main_connector.get_columns(MEASUREMENTS_TABLE))
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
    
    if len(sys.argv) != 2:
        raise Exception("ERROR! Must specify what to update. [images|pointclouds|all]")

    # Parse command line arguments.
    preprocess_pcds = False
    preprocess_jpgs = False
    if sys.argv[1] == "images":
        print("Updating images only...")
        preprocess_jpgs = True
    elif sys.argv[1] == "pointclouds":
        print("Updating pointclouds only...")
        preprocess_pcds = True
    elif sys.argv[1] == "all":
        print("Updating all...")
        preprocess_jpgs = True
        preprocess_pcds = True
                        
    execute_command_preprocess(preprocess_pcds, preprocess_jpgs)
