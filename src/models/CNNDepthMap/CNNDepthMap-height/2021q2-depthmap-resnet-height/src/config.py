from bunch import Bunch

DATASET_MODE_DOWNLOAD = "dataset_mode_download"
DATASET_MODE_MOUNT = "dataset_mode_mount"

CONFIG = Bunch(dict(
    DATASET_MODE=DATASET_MODE_DOWNLOAD,
    DATASET_NAME="anon-depthmap-95k",
    DATASET_NAME_LOCAL="anon-depthmap-mini",
    SPLIT_SEED=0,
    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,
    EPOCHS=1000,
    BATCH_SIZE=16,
    SHUFFLE_BUFFER_SIZE=2560,
    NORMALIZATION_VALUE=7.5,
    LEARNING_RATE=0.0007,
    USE_ONE_CYCLE=True,
    RES_BLOCKS=[2, 5, 5, 2],
    DROPOUTS_LIST=[0.0, 0.0, 0.0, 0.0],
    USE_WANDB=False,
    USE_MULTIGPU=False,

    # Parameters for dataset generation.
    TARGET_INDEXES=[0],  # 0 is height, 1 is weight.
    CODES=['100', '101', '102', '200', '201', '202'],
))
