import sys
# sys.path.append('.')
import dbutils
import pandas as pd
import utils
import numpy as np
from glob2 import glob
from azureml.core import Workspace, Dataset
import yaml
import random
import shutil
import logging
import pickle
import os
import multiprocessing

random.seed(72)

## Load the yaml file
with open("parameters.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile,Loader=yaml.FullLoader)

# Function to extract the qrcode from storagepath of the database
def qrcode(row):
    qrc = row['storage_path']
    split_qrc = qrc.split('/')[1]
    return split_qrc

# Parse all the configuration variables

db_file = cfg["database"]['db_connection_file']
training_file = cfg['csv_paths']['training_paths']
testing_file = cfg['csv_paths']['testing_paths']
number_of_scans =cfg['scans']['scan_amount']
calibration_file = cfg['calibration']['calibration_file']
dataset_name = cfg['data']['dataset']
target_folder = cfg['paths']['target_path']

##connect to the databse
ml_connector = dbutils.connect_to_main_database(db_file)
columns = ml_connector.get_columns('artifacts_with_target')

#query to select the data from the database. NOTE: storing all the data in dataframe and then filtering is much faster
select_artifacts_with_target = "select * from artifacts_with_target;"
database = ml_connector.execute(select_artifacts_with_target, fetch_all=True)
complete_data = database[database['tag'] =='good']
complete_data['qrcode'] = complete_data.apply(qrcode,axis=1)
unique_qrcodes = pd.DataFrame(list(set(complete_data['qrcode'].tolist())),columns =['qrcode'])
usable_qrcodes = unique_qrc[unique_qrc['qrcode'].str.match('15')] ## Select only the relevant qrcodes from the whole database, which is starting with '15' in this case

#Read the already training and testing qrcodes present in the previous dataset
training_qrcodes = pd.read_csv(training_file)
training_qrcodes = training_qrcodes['qrcode'].tolist()
testing_qrcodes = pd.read_csv(training_file)
testing_qrcodes = testing_qrcodes['qrcode'].tolist()
used_qrcodes = training_qrcodes+testing_qrcodes

## filter the qrcodes and sleect the potential qrcodes that can be used for creating new dataset
potential_qrcodes = list(set(usable_qrcodes['qrcode'].tolist()) - set(used_qrcodes))
if len(potential_qrcodes) >= int(number_of_scans):
    selected_qrcodes = random.sample(potential_qrcodes,int(number_of_scans))
else:
    logging.exception("Required qrcodes are more than avaliable qrcodes. Please decrease the numbner of scans or ask for more processed data.")

new_qrcodes = pd.Dataframe(selected_qrcodes.columns =['qrcode'])
new_data = pd.merge(new_qrcodes,complete_data,on='qrcode', how='left') ## merge the filterted qrcodes with all data to obtained all the info on potential qrcodes
# new_data.to_csv('new_data.csv',index=False)

## convert pointcloud to depthmaps
def lenovo_pcd2depth(pcd,calibration):
    points = utils.parsePCD(pcd)
    width = utils.getWidth()
    height = utils.getHeight()
    output = np.zeros((width, height, 1))
    for p in points:
        v = utils.convert3Dto2D(calibration[1], p[0], p[1], p[2])
        x = round(width - v[0] - 1)
        y = round(v[1])
        y = round(height - v[1] - 1)
        if x >= 0 and y >= 0 and x < width and y < height:
            output[x][y] = p[2]
    return output

#Read the Calibration file and set the required shape fro height and width
calibration = utils.parseCalibration(calibration_file)
Width = utils.setWidth(int(240 * 0.75))
Height = utils.setHeight(int(180 * 0.75))

# data = pd.read_csv('new_data.csv')
data = new_data[['qrcode','storage_path','height','weight','key']].values.tolist()

#Mount all the dataset
workspace = Workspace.from_config()
dataset = Dataset.get_by_name(workspace, name= dataset_name)
mount_context = dataset.mount()
mount_context.start()  # this will mount the file streams
print("mounting_point: ", mount_context.mount_point)
source = mount_context.mount_point

print("Processing files:")
unprocess =[]
if not os.path.exists(target):
        os.mkdir(target)

## fucntion to process all the pcd files to depthmaps
def process_file(pointcloud):
    """
    Process the pointcloud files
    Parameters:
    --pointcloud: list containig the qrcode, storagepath of pcd file, height, weight, key
    """
    qrcodefile = pointcloud[0]
    scantype = str(pointcloud[4])
    scanfolder = target+scantype
    if not os.path.exists(scanfolder):
        os.mkdir(scanfolder)
    split_filename = pointcloud[1].split('/')[-1]
    point_file = split_filename.replace('.pcd','.p')
    sourcefile = source+'/'+pointcloud[1]
    targetpath =target+scantype+'/'+qrcodefile
    if not os.path.exists(targetpath):
        os.mkdir(targetpath)
    pickle_file = targetpath+'/'+ point_file
    try:
        sample_depthmap = lenovo_pcd2depth(sourcefile,calibration)
    except:
        unprocess.append(point_file)
        logging.warning("did not process for {}".format(pointcloud))
        pass
    labels = np.array([pointcloud[2],pointcloud[3]])
    data = (sample_depthmap,labels)
    pickle.dump(data, open(pickle_file, "wb"))
    return

proc = multiprocessing.Pool()
for files in datas:
    # launch a process for each file (ish).
    # The result will be approximately one process per CPU core available.
    proc.apply_async(process_file, [files])

p.close()
p.join() # Wait for all child processes to close.

mount_context.stop() ## stop the mounting stream