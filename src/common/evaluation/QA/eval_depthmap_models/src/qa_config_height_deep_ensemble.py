import os

from bunch import Bunch

CONFIG_NAME = os.path.splitext(os.path.basename(__file__))[0]

# Details of model used for evaluation
MODEL_CONFIG = Bunch(dict(
    EXPERIMENT_NAME='q1-ensemble-warmup',
    RUN_IDS=[
        'q1-ensemble-warmup_1610544610_eb44bfe2', 'q1-ensemble-warmup_1610547587_7ca932c3',
        # 'q1-ensemble-warmup_1610547669_5b789bd1', 'q1-ensemble-warmup_1610547705_f2141d0f',
        # 'q1-ensemble-warmup_1610547744_d2b42ce5', 'q1-ensemble-warmup_1610547780_2f000a25',
        # 'q1-ensemble-warmup_1610547816_c3f815df', 'q1-ensemble-warmup_1610547892_8ee6ff49',
        # 'q1-ensemble-warmup_1610547928_b9519b6a', 'q1-ensemble-warmup_1610547986_ad0186b8',
        # 'q1-ensemble-warmup_1610548023_99ac6060', 'q1-ensemble-warmup_1610548064_afefd4e4',
        # 'q1-ensemble-warmup_1610548106_69993d24', 'q1-ensemble-warmup_1610548137_a8c52d63',
        # 'q1-ensemble-warmup_1610548168_914ce1f6', 'q1-ensemble-warmup_1610548209_9692a253',
    ],
    INPUT_LOCATION='outputs',
    NAME='best_model.ckpt',
))


EVAL_CONFIG = Bunch(dict(
    # Name of evaluation
    NAME='q3-depthmap-plaincnn-height-95k-run_03',

    # Experiment in Azure ML which will be used for evaluation
    EXPERIMENT_NAME="QA-pipeline",
    CLUSTER_NAME="gpu-cluster",

    # Used for Debug the QA pipeline
    DEBUG_RUN=False,

    # Will run eval on specified # of scan instead of full dataset
    DEBUG_NUMBER_OF_SCAN=5,

    SPLIT_SEED=0,
))

# Details of Evaluation Dataset
DATA_CONFIG = Bunch(dict(
    NAME='anon-realtime-testdata',  # Name of evaluation dataset

    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,

    BATCH_SIZE=512,  # Batch size for evaluation
    NORMALIZATION_VALUE=7.5,

    TARGET_INDEXES=[0, 5],  # 0 is height, 1 is weight.
    CODES=['100', '101', '102', '200', '201', '202']
))


# Result configuration for result generation after evaluation is done
RESULT_CONFIG = Bunch(dict(
    # Error margin on various ranges
    #EVALUATION_ACCURACIES = [.2, .4, .8, 1.2, 2., 2.5, 3., 4., 5., 6.]
    ACCURACIES=[.2, .4, .6, 1, 1.2, 2., 2.5, 3., 4., 5., 6.],  # 0.2cm, 0.4cm, 0.6cm, 1cm, ...
    ACCURACY_MAIN_THRESH=1.0,
    COLUMNS=['qrcode', 'artifact', 'scantype', 'GT', 'predicted'],

    USE_UNCERTAINTY=True,  # Flag to enable model uncertainty calculation
    UNCERTAINTY_THRESHOLD_IN_CM=4.,

    TARGET_INDEXES=[0],
    # path of csv file in the experiment which final result is stored
    SAVE_PATH=f'./outputs/{CONFIG_NAME}',
))
