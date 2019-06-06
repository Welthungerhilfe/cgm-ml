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


def main():

    # Check the arguments.
    if len(sys.argv) != 2:
        raise Exception("ERROR! Must provide model filename.")
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        raise Exception("ERROR! \"{}\" does not exist.".format(model_path))
    if not os.path.isfile(model_path):
        raise Exception("ERROR! \"{}\" is not a file.".format(model_path))

    # Get the training QR-codes.
    search_path = os.path.join(os.path.dirname(model_path), "*.p")
    paths = glob.glob(search_path)
    details_path = [path for path in paths if "details" in path][0]
    details = pickle.load(open(details_path, "rb"))
    qrcodes_train = details["qrcodes_train"]
        
    # Create database connection.
    db_connector = dbutils.connect_to_main_database()

    # Query the database for artifacts.
    print("Getting all artifacts...")
    sql_statement = ""
    # Select all artifacts.
    sql_statement += "SELECT pointcloud_data.id, pointcloud_data.path, measurements.height_cms, pointcloud_data.qrcode FROM pointcloud_data"
    # Join them with measurements.
    sql_statement += " INNER JOIN measurements ON pointcloud_data.measurement_id=measurements.id"
    # Only take into account manual measurements.
    sql_statement += " WHERE measurements.type=\'manual\'"
    artifacts = db_connector.execute(sql_statement, fetch_all=True)
    print("Found {} artifacts.".format(len(artifacts)))

    # Method for processing a set of artifacts.
    # Note: This method will run in its own process.
    def process_artifacts(artifacts):
        
         # Create database connection.
        db_connector = dbutils.connect_to_main_database()
    
        # Load the model first.
        model = load_model(model_path)
        model_name = model_path.split("/")[-2]
        
        # Evaluate and create SQL-statements.
        bar = progressbar.ProgressBar(max_value=len(artifacts))
        for artifact_index, artifact in enumerate(artifacts):
            bar.update(artifact_index)

            # Execute SQL statement.
            try:
                # Load the artifact and evaluate.
                artifact_id, pcd_path, target_height, qrcode = artifact
                pcd_array = utils.load_pcd_as_ndarray(pcd_path)
                pcd_array = utils.subsample_pointcloud(pcd_array, 10000)
                mse, mae = model.evaluate(np.expand_dims(pcd_array, axis=0), np.array([target_height]), verbose=0)
                if qrcode in qrcodes_train:
                    misc = "training"
                else:
                    misc = "nottraining"
                
                # Create an SQL statement.
                sql_statement = ""
                sql_statement += "INSERT INTO artifact_quality (type, key, value, artifact_id, misc)"
                sql_statement += " VALUES(\'{}\', \'{}\', \'{}\', \'{}\', \'{}\');".format(model_name, "mae", mae, artifact_id, misc)

                # Call database.
                result = db_connector.execute(sql_statement)
            except psycopg2.IntegrityError:
                print("Already in DB. Skipped.", pcd_path)
            except ValueError:
                print("Skipped.", pcd_path)
        bar.finish()

    # Run this in multiprocess mode.
    utils.multiprocess(artifacts, process_method=process_artifacts, process_individial_entries=False, progressbar=False)
    print("Done.")
        
        
def load_model(model_path):

    input_shape = (10000, 3)
    output_size = 1
    model = modelutils.create_point_net(input_shape, output_size, hidden_sizes = [512, 256, 128])
    model.load_weights(model_path)
    model.compile(
        optimizer="rmsprop",
        loss="mse",
        metrics=["mae"]
    )
    return model 


if __name__ == "__main__":
    main()