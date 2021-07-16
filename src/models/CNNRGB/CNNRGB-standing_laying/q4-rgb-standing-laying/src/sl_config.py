from bunch import Bunch

DATASET_MODE_DOWNLOAD = "dataset_mode_download"
DATASET_MODE_MOUNT = "dataset_mode_mount"

CONFIG = Bunch(dict(
    DATASET_MODE=DATASET_MODE_DOWNLOAD,
    DATASET_NAME='anon-rgb-classification',
    DATASET_NAME_LOCAL="anon-rgb-classification",
    SPLIT_SEED=0,
    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,
    EPOCHS=15,
    BATCH_SIZE=256,
    SHUFFLE_BUFFER_SIZE=2560,
    NORMALIZATION_VALUE=256,
    LEARNING_RATE=1e-5,
    TUNE_EPOCHS=10,
    USE_MULTIGPU=False,
    CLUSTER_NAME='gpu-cluster',
))
