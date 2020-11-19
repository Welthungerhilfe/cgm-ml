class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DATASET_MODE_DOWNLOAD = "dataset_mode_download"
DATASET_MODE_MOUNT = "dataset_mode_mount"

CONFIG = dotdict(dict(
    DATASET_MODE=DATASET_MODE_DOWNLOAD,
    DATASET_NAME="anon-depthmap-95k",
    DATASET_NAME_LOCAL="anon-depthmap-mini",
    SPLIT_SEED=0,
    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,
    EPOCHS=500,
    BATCH_SIZE=256,
    SHUFFLE_BUFFER_SIZE=2560,
    NORMALIZATION_VALUE=7.5,
    LEARNING_RATE=0.0007,
    USE_ONE_CYCLE=True,
    WEIGHT_DECAY=0.01,  # weightDecay=0.1 (often best), 0.01 (default in fastai)

    # Parameters for dataset generation.
    TARGET_INDEXES=[0],  # 0 is height, 1 is weight.
))
