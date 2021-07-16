from bunch import Bunch

DATA_AUGMENTATION_SAME_PER_CHANNEL = "same_per_channel"
DATA_AUGMENTATION_DIFFERENT_EACH_CHANNEL = "different_each_channel"
DATA_AUGMENTATION_NO = "no"

SAMPLING_STRATEGY_SYSTEMATIC = "systematic"
SAMPLING_STRATEGY_WINDOW = "window"

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
    EPOCHS_TUNE=200,
    BATCH_SIZE=64,
    SHUFFLE_BUFFER_SIZE=2560,
    NORMALIZATION_VALUE=7.5,
    LEARNING_RATE=0.01,
    LEARNING_RATE_TUNE=0.001,

    # Parameters for dataset generation.
    TARGET_INDEXES=[0],  # 0 is height, 1 is weight.
    N_ARTIFACTS=5,  # number of artifact in one sample of multiartifact approach
    CODES=('100', '101', '102', '200', '201', '202'),
    N_REPEAT_DATASET=1,
    DATA_AUGMENTATION_MODE=DATA_AUGMENTATION_NO,
    SAMPLING_STRATEGY=SAMPLING_STRATEGY_SYSTEMATIC,
    USE_DROPOUT=False,
    USE_ONE_CYCLE=False,
    USE_MULTIGPU=False,
    CLUSTER_NAME='gpu-cluster',

    PRETRAINED_RUN="q3-depthmap-plaincnn-height-95k_1610709869_2e00a6ef",  # Run4 (baseline: min(val_mae)=1.78cm)

    SHOULD_FREEZE_BASE=True,
))

CONFIG.PRETRAINED_EXPERIMENT = "_".join(CONFIG.PRETRAINED_RUN.split('_')[:-2])
