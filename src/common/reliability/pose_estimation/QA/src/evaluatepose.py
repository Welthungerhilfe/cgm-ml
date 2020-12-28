import os
import random
import sys
import time

import cv2
import glob2 as glob
import pandas as pd
import tensorflow as tf
from azureml.core import Experiment, Workspace
from azureml.core.run import Run

import posepoints
import utils
from config import EVAL_CONFIG, DATA_CONFIG, RESULT_CONFIG
from constants import REPO_DIR


def init(proto_txt, model_name):
    global net
    print('proto ', proto_txt)
    net = cv2.dnn.readNetFromCaffe(proto_txt, model_name)
    return net


if __name__ == "__main__":
    # Make experiment reproducible
    tf.random.set_seed(EVAL_CONFIG.SPLIT_SEED)
    random.seed(EVAL_CONFIG.SPLIT_SEED)

    # Get the current run.
    run = Run.get_context()

    # Offline run. Download the sample dataset and run locally. Still push results to Azure.
    if run.id.startswith("OfflineRun"):
        print("Running in offline mode...")

        # Access workspace.
        print("Accessing workspace...")
        workspace = Workspace.from_config()
        experiment = Experiment(workspace, EVAL_CONFIG.EXPERIMENT_NAME)
        run = experiment.start_logging(outputs={}, snapshot_directory={})

        # Get dataset.
        print("Accessing dataset...")
        dataset_name = DATA_CONFIG.NAME
        dataset_path = str(REPO_DIR / "data" / dataset_name)
        if not os.path.exists(dataset_path):
            dataset = workspace.datasets[dataset_name]
            dataset.download(target_path=dataset_path, overwrite=False)

    # Online run. Use dataset provided by training notebook.
    else:
        print("Running in online mode...")
        experiment = run.experiment
        workspace = experiment.workspace
        dataset_path = run.input_datasets["dataset"]

    # Get the QR-code paths.
    dataset_path = os.path.join(dataset_path, "scans")
    print("Dataset path:", dataset_path)
    print("Getting QR code paths...")
    qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
    print("QR code paths: ", len(qrcode_paths))
    assert len(qrcode_paths) != 0

    if EVAL_CONFIG.DEBUG_RUN and len(qrcode_paths) > EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN:
        qrcode_paths = qrcode_paths[:EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN]
        print("Executing on {} qr codes for FAST RUN".format(EVAL_CONFIG.DEBUG_NUMBER_OF_SCAN))

    # Shuffle and take approximately 1/6ths of the data for pose estimation
    random.shuffle(qrcode_paths)
    split_index = int(len(qrcode_paths) * 0.17)
    qrcode_paths_pose_est = qrcode_paths[:split_index]
    print('qrcode_paths_pose_est len- ', len(qrcode_paths_pose_est))

    # Get the RGBs.
    print("Getting RGB paths...")
    rgb_files = utils.get_rgb_files(qrcode_paths_pose_est)
    del qrcode_paths
    del qrcode_paths_pose_est

    print("Using {} rgb files for pose estimation.".format(len(rgb_files)))

    qrcode_list, artifact_list = utils.get_column_list(rgb_files)

    num_scan_files = DATA_CONFIG.NUM_SCANFILES
    if num_scan_files == 0:
        num_scan_files = len(rgb_files)
    print('num_scan_files - ', num_scan_files)

    df = pd.DataFrame({'artifact': ''}, index=[1], columns=RESULT_CONFIG.COLUMNS)

    proto = DATA_CONFIG.PROTOTXT_PATH
    model = DATA_CONFIG.MODELTYPE_PATH
    datasetType = DATA_CONFIG.DATASETTYPE_PATH
    print('proto ', proto)
    print('model ', model)
    print(f"datasetType {datasetType}")

    # set up the network with the prototype and model
    net = init(proto, model)

    # get POSE details
    dataset_type_and_model, body_parts, pose_pairs = posepoints.set_pose_details(datasetType)

    # Add the other columns
    df, columns = posepoints.add_columns_to_dataframe(body_parts, pose_pairs, df)

    print('df.columns ', df.columns)

    # pose estimation points
    z = 0
    # dataframe
    df = pd.DataFrame(columns=df.columns)
    df.set_index('artifact')

    artifacts = []
    for j in range(num_scan_files):
        artifact = utils.get_file_name(rgb_files[j])
        artifacts.append(artifact)

    errors = []
    not_processed = []
    processed_len = 0
    start_t = time.time()
    for i in range(num_scan_files):
        artifact = artifacts[i]
        points = None

        try:
            imagePath = rgb_files[i]
            points = posepoints.pose_estimate(imagePath, net, body_parts, pose_pairs,
                                              width=250, height=250)
            z = z + 1

            # set artifact name
            df.loc[z, "artifact"] = artifact

            for key, value in zip(columns, points):
                df.loc[z, key] = value

        except Exception:
            e = sys.exc_info()[0]
            errors.append(e)
            not_processed.append(artifact)

    dfErrors = None
    not_process_len = 0
    if len(not_processed) > 0:
        not_process_len = len(not_processed)
        values = [list(row) for row in zip(not_processed, errors)]
        dfErrors = pd.DataFrame(values, columns=['artifact', 'error'])
        dfErrors.set_index('artifact')
        print(f"Not processed {not_process_len} scans, for feature extraction.")
        err_path = "dfRgbPoseEst_errors.json"

        # create folder if need be
        if not os.path.exists('outputs'):
            os.makedirs('outputs', mode=0o777, exist_ok=False)
        # write the file
        dfErrors.to_json(f"outputs/{err_path}", index=True)

    processed_len = num_scan_files - not_process_len
    print(f"Total time for {processed_len} scans, pose estimation is {time.time() - start_t}.")

    print(df.head())
    print('df.shape', df.shape)
    # save pose estimation results to file
    # OUTFILE_PATH = f"{EVAL_CONFIG.EXPERIMENT_NAME}_pose_points.csv"
    # df.to_csv(f"outputs/{OUTFILE_PATH}", index=True)

    # write as json, instead, easy to read
    OUTFILE_PATH = f"{EVAL_CONFIG.EXPERIMENT_NAME}_pose_points.json"
    df.to_json(f"outputs/{OUTFILE_PATH}", index=True)

    # Done.
    run.complete()
