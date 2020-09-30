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
import os
import glob2 as glob
import dbutils
import progressbar
import time
import datetime
import cv2


whhdata_path = "/whhdata"
media_subpath = "person"

main_connector = dbutils.connect_to_main_database()


def execute_command_updatemedia(update_jpgs=False, update_pcds=True):
    print("Updating media...")
    
    # Process JPGs.
    if update_jpgs == True:
        table = "image_data"
        glob_search_path = os.path.join(whhdata_path, media_subpath, "**/*.jpg")
        print("Searching at {}... This might take a while!".format(glob_search_path))
        jpg_paths = glob.glob(glob_search_path) # TODO make this work again!
        #jpg_paths = ["/whhdata/person/MH_WHH_0153/measurements/1537860868501/rgb/rgb_MH_WHH_0153_1537860868501_104_95405.92970875901.jpg"]
        print("Found {} JPGs.".format(len(jpg_paths)))
        update_media_table(jpg_paths, table, get_image_values)
    
    # Process PCDs.
    if update_pcds == True:
        table = "pointcloud_data"
        glob_search_path = os.path.join(args.path, media_subpath, "**/*.pcd")
        print("Searching at {}... This might take a while!".format(glob_search_path))
        pcd_paths = glob.glob(glob_search_path)
        #pcd_paths = ["/whhdata/person/MH_WHH_0030/measurements/1536913928288/pc/pc_MH_WHH_0030_1536913928288_104_000.pcd"]
        print("Found {} PCDs.".format(len(pcd_paths)))
        update_media_table(pcd_paths, table, get_pointcloud_values, batch_size=100)

    
def update_media_table(file_paths, table, get_values, batch_size=1000):
    insert_count = 0
    no_measurements_count = 0
    skip_count = 0
    bar = progressbar.ProgressBar(max_value=len(file_paths))
    sql_statement = ""
    last_index = len(file_paths) - 1
    for index, file_path in enumerate(file_paths):
        bar.update(index)
        
        # Check if there is already an entry.
        path = os.path.basename(file_path)
        sql_statement_select = dbutils.create_select_statement(table, ["path"], [file_path])
        results = main_connector.execute(sql_statement_select, fetch_all=True)
  
        # No results found. Insert.
        if len(results) == 0:
            insert_data = { "path": path }
            default_values = get_default_values(file_path, table)
            if default_values != None:
                insert_data.update(default_values)
                insert_data.update(get_values(file_path))
                sql_statement += dbutils.create_insert_statement(table, insert_data.keys(), insert_data.values())
                insert_count += 1
            else:
                no_measurements_count += 1
        
        # Found a result. Update.
        elif len(results) != 0:
            # TODO check if measurement id is missing or not
            skip_count += 1
        
        # Update database.
        if index != 0 and ((index % batch_size) == 0) or index == last_index:
            if sql_statement != "":
                #print("")
                #print(sql_statement)
                #print("")
                result = main_connector.execute(sql_statement)
                sql_statement = ""
   
    bar.finish()
    print("Inserted {} new entries.".format(insert_count))
    print("No measurements for {} entries.".format(no_measurements_count))
    print("Skipped {} entries.".format(skip_count))
    
    
def get_default_values(path, table):
    
    # Split and check the path.
    path_split = path.split("/")
    assert path_split[1] == whhdata_path[1:]
    assert path_split[2] == media_subpath
    
    # Get important values from path.
    qrcode = path_split[3]
    timestamp = path_split[-1].split("_")[-3]
    
    # Getting timestamp.
    last_updated, _ = get_last_updated()

    # Get id of measurement.
    threshold = int(60 * 60 * 24 * 1000)
    sql_statement = dbutils.create_select_statement("measurements", ["qrcode"], [qrcode])
    sql_statement = ""
    sql_statement += "SELECT id"
    sql_statement += " FROM measurements WHERE"
    sql_statement += " qrcode = '{}'".format(qrcode)
    sql_statement += " AND type = 'manual'"
    sql_statement += " AND ABS(timestamp - {}) < {}".format(timestamp, threshold)
    sql_statement += ";"
    results = main_connector.execute(sql_statement, fetch_all=True)
    
    # Prepare values.
    values = {}
    values["path"] = path
    values["qrcode"] = qrcode
    values["last_updated"] = last_updated
    values["rejected_by_expert"] = False

    # Measurement id not found.
    if len(results) == 0:
        print("No measurement_id found for {}".format(path))
        
    # Found a measurement id.
    else:
        values["measurement_id"] = results[0][0]
    
    return values
    
    
def get_last_updated():
    last_updated = time.time()
    last_updated_readable = datetime.datetime.fromtimestamp(last_updated).strftime('%Y-%m-%d %H:%M:%S')
    return last_updated, last_updated_readable
    
    
def get_pointcloud_values(path):
    number_of_points = 0
    confidence_min = 0.0
    confidence_avg = 0.0
    confidence_std = 0.0
    confidence_max = 0.0
    
    centroid_x = 0.0
    centroid_y = 0.0
    centroid_z = 0.0
    
    stdev_x = 0.0
    stdev_y = 0.0
    stdev_z = 0.0
    
    error = False
    error_message = ""
    
    try:
        pointcloud = utils.load_pcd_as_ndarray(path)
        number_of_points = len(pointcloud)
        confidence_min = float(np.min(pointcloud[:,3]))
        confidence_avg = float(np.mean(pointcloud[:,3]))
        confidence_std = float(np.std(pointcloud[:,3]))
        confidence_max = float(np.max(pointcloud[:,3]))
        
        centroid_x = float(np.mean(pointcloud[:,0]))
        centroid_y = float(np.mean(pointcloud[:,1]))
        centroid_z = float(np.mean(pointcloud[:,2]))
        
        stdev_x = float(np.mean(pointcloud[:,0]))
        stdev_y = float(np.mean(pointcloud[:,1]))
        stdev_z = float(np.mean(pointcloud[:,2]))
        
    except Exception as e:
        print("\n", path, e)
        error = True
        error_message = str(e)
    except ValueError as e:
        print("\n", path, e)
        error = True
        error_message = str(e)
    
    values = {}
    values["number_of_points"] = number_of_points
    values["confidence_min"] = confidence_min
    values["confidence_avg"] = confidence_avg
    values["confidence_std"] = confidence_std
    values["confidence_max"] = confidence_max
    values["centroid_x"] = centroid_x
    values["centroid_y"] = centroid_y
    values["centroid_z"] = centroid_z
    values["stdev_x"] = stdev_x
    values["stdev_y"] = stdev_y
    values["stdev_z"] = stdev_z
    values["had_error"] = error
    values["error_message"] = error_message
    return values


def get_image_values(path):
    width = 0.0
    height = 0.0
    blur_variance = 0.0
    error = False
    error_message = ""
    try:
        image = cv2.imread(path)
        width = image.shape[0]
        height = image.shape[1]
        blur_variance = get_blur_variance(image)
    except Exception as e:
        print("\n", path, e)
        error = True
        error_message = str(e)
    except ValueError as e:
        print("\n", path, e)
        error = True
        error_message = str(e)

    values = {}
    values["width_px"] = width
    values["height_px"] = height
    values["blur_variance"] = blur_variance
    values["has_face"] = False # TODO fix
    values["had_error"] = error
    values["error_message"] = error_message
    return values
    
    
def get_blur_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        raise Exception("ERROR! Must specify what to update. [images|pointclouds|all]")

    # Parse command line arguments.
    update_jpgs = False
    update_pcds = False
    if sys.argv[1] == "images":
        print("Updating images only...")
        update_jpgs = True
    elif sys.argv[1] == "pointclouds":
        print("Updating pointclouds only...")
        update_pcds = True
    elif sys.argv[1] == "all":
        print("Updating all...")
        update_jpgs = True
        update_pcds = True
                        
    execute_command_updatemedia(update_jpgs, update_pcds)
                        
                        
                        
                        
                        
    