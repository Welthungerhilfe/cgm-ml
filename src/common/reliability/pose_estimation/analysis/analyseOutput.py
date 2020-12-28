from src import posepoints
from src.config import DATA_CONFIG
import json
from collections import defaultdict
from statistics import mean, stdev

import pandas as pd
import sys
sys.path.append("..")


def load_param_config():
    """
    Loading configuration file containing Model and Dataset type
    """
    ds = DATA_CONFIG.DATASETTYPE_PATH
    model = DATA_CONFIG.MODELTYPE_PATH
    print("Dataset type = {}\n Model = {} ".format(ds, model))
    return ds

def initialisation(dataset="default-dataset"):
    """
    Initialising an empty POSE_PAIR dictionary according to the
    dataset type used for training the model.
    During analysis, if a pose pair is found missing for an image
    then the value corresponding to the pose pair is incremented
    and saved in this dictionary
    """
    if dataset in 'COCO':
        poses = ['P12', 'P15', 'P23', 'P34', 'P56',
                 'P67', 'P18', 'P89', 'P910', 'P111',
                 'P1112', 'P1213', 'P10', 'P014',
                 'P1416', 'P015', 'P1517']
        # create a dict with pose_pair as keys and initialize
        # all the value to 0.
        pose_pair = dict.fromkeys(poses, 0)
        print("Initialize an empty POSE_PAIR dictionary:\n{}\n".
              format(pose_pair))
    elif dataset in 'MPI':
        poses = ['P01', 'P12', 'P23', 'P34', 'P15', 'P56', 'P67',
                 'P114', 'P148', 'P89', 'P910',
                 'P1411', 'P1112', 'P1213']
        # create a dict with pose_pair as keys and initialize
        # all the value to 0
        pose_pair = dict.fromkeys(poses, 0)
        print("Initialize an empty POSE_PAIR dictionary:\n{}\n".
              format(pose_pair))
    else:  # default dataset
        poses = ['P18', 'P12', 'P15', 'P23', 'P34', 'P56', 'P67',
                 'P89', 'P910', 'P1011', 'P812', 'P1213', 'P1314',
                 'P10', 'P015', 'P1517', 'P016', 'P1618', 'P217',
                 'P518', 'P1419', 'P1920', 'P1421', 'P1122',
                 'P2223', 'P1124']
        # create a dict with pose_pair as keys and initialize all
        # the value to 0
        pose_pair = dict.fromkeys(poses, 0)
        print("Initialize an empty POSE_PAIR dictionary:\n{}\n".
              format(pose_pair))
    return pose_pair

def set_pose_pair_body_parts(dataset_typ, num_qrcodes):
    """
    This function is called to set pose details such as POSE_PAIRS
    and BODY_PARTS
    """
    dataset_type_model, BODY_PARTS, POSE_PAIRS = \
        posepoints.setPoseDetails(dataset_typ)

    print("Total no. of images used during training: ", num_qrcodes)
    print("BODY_PARTS present in a training image:\n{}\n".
          format(BODY_PARTS))
    df = pd.DataFrame({
        'artifact': ''
    }, index=[1], columns=['artifact'])
    df, columns = posepoints.addColumnsToDataframe(BODY_PARTS,
                                                   POSE_PAIRS,
                                                   df)
    print("POSE_PAIRS that connect BODY_PARTS:{}".format(columns))
    l: int = len(columns)
    print("Length of the POSE_PAIR list:{}\n".format(l))

def load_json():
    """
    Load the training result of the Pose estimation model
    """
    with open("pose_estimation_output.json", "r") as f:
        data = json.load(f)
    # The experiment is ran on 1/6th (107229) of the total
    # number of RGB images in the dataset anon_rgb_training
    # (643,374)
    num_of_artifacts = len(data['artifact'])
    print("No. of artifacts = ", num_of_artifacts)
    return data, num_of_artifacts

def analyse(data, pose_pair, num_qrcodes):
    """
    Parse anonrgbtrain_poseestimation_ps_posepoints.json file to
    detect missing posepoints for each POSE_PAIR in all the images
    """
    missing_posepair = []
    missing_image_posepair = defaultdict(list)
    missing_posepair_with_count = {}
    for k1, v1 in data.items():
        for k2, v2 in v1.items():
            if not v2:
                # If a POSE_PAIR is not found add it to the
                # missing_posepair list
                missing_posepair.append(k1)
                missing_image_posepair[k2].append(k1)

                if k1 in pose_pair.keys():
                    # Increment the value of the corresponding key in
                    # the POSE_PAIR dict
                    pose_pair[k1] = pose_pair[k1] + 1
                else:
                    print("All POSE_POINTS detected in all the images"
                          " in the dataset")
                # print("pose_pair",pose_pair)
                for k, v in pose_pair.items():
                    if v != 0:
                        missing_posepair_with_count[k] = v
    print("POSE_PAIR and the corresponding no. of undetected "
          "POSE_POINTS from {} images: {}\n".
          format(num_qrcodes, missing_posepair_with_count))
    num_posepoints_missed = len(missing_posepair)
    print("\nTotal number of undetected POSE_POINTS in all images = {}".
          format(num_posepoints_missed))
    # print("Image and its corresponding POSE_PAIR that is undetected ",
    # missing_image_posepair)
    num_missed_images = len(missing_image_posepair.keys())
    print("\nOut of {} images, in {} images, POSE_POINTS have not "
          "been detected".
          format(num_qrcodes, num_missed_images))

    dataset_len = num_qrcodes
    accuracy = (num_missed_images / dataset_len) * 100
    print("\n###############EVALUATION################")
    print("\nACCURACY of CAFFE Pose estimation model = {}%".
          format(accuracy))
    max_value = max(missing_posepair_with_count.values())
    max_keys = [k for k, v in missing_posepair_with_count.items() if
                v == max_value]
    print("POSE_PAIR {} has the maximum number of undetected "
          "pose points = {}".
          format(max_keys, max_value))
    min_value = min(missing_posepair_with_count.values())
    min_keys = [k for k, v in missing_posepair_with_count.items() if
                v == min_value]
    print("POSE_PAIR {} has the minimum number of undetected "
          "pose points = {}".
          format(min_keys, min_value))
    mean_values = mean(missing_posepair_with_count.values())
    stdev_values = stdev(missing_posepair_with_count.values())
    # print("Total sum of undetected pose_points = ",sum_values)
    print("\nMean of undetected pose_points = ", mean_values)
    print("Standard deviation of undetected pose_points = {}\n ".
          format(stdev_values))

    if __name__ == "__main__":
        dataset_type = load_param_config()
        posepair = initialisation(dataset_type)
        data_to_analyse, num_images = load_json()
        set_pose_pair_body_parts(dataset_type, num_images)
        analyse(data_to_analyse, posepair, num_images)
