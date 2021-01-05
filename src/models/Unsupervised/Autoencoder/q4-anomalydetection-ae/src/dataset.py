import os
import random
from pathlib import Path

import glob2 as glob
import tensorflow as tf
import tensorflow_datasets as tfds
from tmp_model_util.utils import download_dataset, get_dataset_path

from config import DATASET_MODE_DOWNLOAD, DATASET_MODE_MOUNT
from constants import DATA_DIR_ONLINE_RUN, REPO_DIR


def create_datasets(workspace, experiment, run, offline_run, CONFIG):

    DATA_DIR = REPO_DIR / 'data' if offline_run else Path(".")
    print(f"DATA_DIR: {DATA_DIR}")

    # Offline run. Download the sample dataset and run locally. Still push results to Azure.
    if offline_run:
        dataset_name = CONFIG.DATASET_NAME_LOCAL
        dataset_path = get_dataset_path(DATA_DIR, dataset_name)
        download_dataset(workspace, dataset_name, dataset_path)

    # Online run. Use dataset provided by training notebook.
    else:
        dataset_name = CONFIG.DATASET_NAME

        # Mount or download
        if CONFIG.DATASET_MODE == DATASET_MODE_MOUNT:
            dataset_path = run.input_datasets["dataset"]
        elif CONFIG.DATASET_MODE == DATASET_MODE_DOWNLOAD:
            dataset_path = get_dataset_path(DATA_DIR_ONLINE_RUN, dataset_name)
            download_dataset(workspace, dataset_name, dataset_path)
        else:
            raise NameError(f"Unknown DATASET_MODE: {CONFIG.DATASET_MODE}")

    print(f"Using dataset path {dataset_path}")

    # Branch into specific datasets.
    if dataset_name == "anon_rgb_training":
        return __create_anon_rgb_training(dataset_path, CONFIG)
    elif dataset_name == "anomaly_detection_data":
        return __create_anomaly_detection_data(dataset_path, CONFIG)
    else:
        assert False, f"Unexpected dataset {dataset_name}."


def __create_anon_rgb_training(dataset_path, CONFIG):

    # Get the QR-code paths.
    dataset_path = os.path.join(dataset_path, "scans")
    print("Dataset path:", dataset_path)
    #print(glob.glob(os.path.join(dataset_path, "*"))) # Debug
    print("Getting QR-code paths...")
    qrcode_paths = glob.glob(os.path.join(dataset_path, "*"))
    print("qrcode_paths: ", len(qrcode_paths))
    assert len(qrcode_paths) != 0

    # Restricting to a fixed number of scans.
    print("Slicing scans...")
    qrcode_paths = qrcode_paths[:CONFIG.DATASET_MAX_SCANS]
    print("qrcode_paths: ", len(qrcode_paths))
    #assert len(qrcode_paths) == CONFIG.DATASET_MAX_SCANS

    # Shuffle and split into train and validate.
    random.shuffle(qrcode_paths)
    split_index = int(len(qrcode_paths) * 0.8)
    qrcode_paths_training = qrcode_paths[:split_index]
    qrcode_paths_validate = qrcode_paths[split_index:]

    del qrcode_paths

    # Show split.
    print("Paths for training:")
    print(len(qrcode_paths_training))
    print("\t" + "\n\t".join(qrcode_paths_training))
    print("Paths for validation:")
    print(len(qrcode_paths_validate))
    print("\t" + "\n\t".join(qrcode_paths_validate))
    assert len(qrcode_paths_training) > 0 and len(qrcode_paths_validate) > 0

    # For mapping scan paths to samples.
    # Files are shuffled per scan.
    # And restricted to a maximum of samples per scan.
    def get_files(scan_paths, extension="jpg"):
        result_paths = []
        for scan_path in scan_paths:
            sample_paths = glob.glob(os.path.join(scan_path, "**", "*." + extension))
            random.shuffle(sample_paths)
            sample_paths = sample_paths[:CONFIG.DATASET_MAX_SAMPLES_PER_SCAN]
            result_paths.extend(sample_paths)
        return result_paths

    # Get the samples.
    print("Getting paths...")
    paths_training = get_files(qrcode_paths_training)
    paths_validate = get_files(qrcode_paths_validate)
    print("Using {} files for training.".format(len(paths_training)))
    print("Using {} files for validation.".format(len(paths_validate)))
    assert paths_training != 0
    assert paths_validate != 0

    # Shuffle all.
    random.shuffle(paths_training)
    random.shuffle(paths_validate)

    # QR codes not necessary anymore.
    del qrcode_paths_training
    del qrcode_paths_validate

    # Function for loading and processing images.
    def tf_load_sample(path):
        def py_load_sample(path):
            image_string = tf.io.read_file(path)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)
            image = tf.image.resize(image, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH))
            image = tf.image.rot90(image, k=3)
            image = image / 255.0
            return image

        image = tf.py_function(py_load_sample, [path], [tf.float32])[0]
        image.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 3))
        return image

    # Create dataset for training.
    # Note: Model will do caching et cetera.
    paths = paths_training
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset_norm = dataset.map(lambda path: tf_load_sample(path))
    dataset_train = dataset_norm
    del dataset_norm

    # Create dataset for validation.
    # Note: No shuffle necessary.
    # Note: Model will do caching et cetera.
    paths = paths_validate
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset_norm = dataset.map(lambda path: tf_load_sample(path))
    dataset_validate = dataset_norm
    del dataset_norm

    # Create dataset for anomaly detection.
    # Note: Model will do caching et cetera.
    def tf_preprocess(image):
        image = tf.image.resize(image, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH)) / 255.0
        return image

    dataset_anomaly = tfds.load("cats_vs_dogs", split="train[:10%]")
    dataset_anomaly = dataset_anomaly.map(lambda sample: tf_preprocess(sample["image"]))

    # Done.
    return dataset_train, dataset_validate, dataset_anomaly


def __create_anomaly_detection_data(dataset_path, CONFIG):

    #dataset_path = os.path.join(dataset_path, "scans")
    print(glob.glob(os.path.join(dataset_path, "*")))

    good_paths = glob.glob(os.path.join(dataset_path, "not_bad", "*.jpg"))
    bad_paths = glob.glob(os.path.join(dataset_path, "bad", "*.jpg"))

    random.shuffle(good_paths)
    split_index = int(0.8 * len(good_paths))
    paths_training = good_paths[:split_index]
    paths_validate = good_paths[split_index:]
    paths_anomaly = bad_paths

    print(f"Training paths: {len(paths_training)}")
    print(f"Validate paths: {len(paths_validate)}")
    print(f"anomaly paths: {len(paths_anomaly)}")
    assert len(paths_training) * len(paths_validate) * len(paths_anomaly) != 0

    # Function for loading and processing images.
    def tf_load_sample(path):
        def py_load_sample(path):
            image_string = tf.io.read_file(path)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)
            image = tf.image.resize(image, (CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH))
            image = tf.image.rot90(image, k=3)
            image = image / 255.0
            return image

        image = tf.py_function(py_load_sample, [path], [tf.float32])[0]
        image.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 3))
        return image

    # Create dataset for training.
    # Note: Model will do caching et cetera.
    paths = paths_training
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset_norm = dataset.map(lambda path: tf_load_sample(path))
    dataset_train = dataset_norm
    del dataset_norm

    # Create dataset for validation.
    # Note: No shuffle necessary.
    # Note: Model will do caching et cetera.
    paths = paths_validate
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset_norm = dataset.map(lambda path: tf_load_sample(path))
    dataset_validate = dataset_norm
    del dataset_norm

    # Create dataset for validation.
    # Note: No shuffle necessary.
    # Note: Model will do caching et cetera.
    paths = paths_anomaly
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset_norm = dataset.map(lambda path: tf_load_sample(path))
    dataset_anomaly = dataset_norm
    del dataset_norm

    return dataset_train, dataset_validate, dataset_anomaly
