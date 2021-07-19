import os

from bunch import Bunch

CONFIG_NAME = os.path.splitext(os.path.basename(__file__))[0]

# Details of model used for evaluation
MODEL_CONFIG = Bunch(dict(
    EXPERIMENT_NAME='q4-depthmap-plaincnn-weight-95k',
    RUN_ID='q4-depthmap-plaincnn-weight-95k_1611336518_642a9c58',  # Run 2
    INPUT_LOCATION='outputs',
    NAME='best_model.ckpt',
))


EVAL_CONFIG = Bunch(dict(
    # Name of evaluation
    NAME='q4-depthmap-plaincnn-weight-95k_run_02',

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

    # 0 is height, 1 is weight, 2 is muac, 3 is age,
    # 4 is sex('male' or 'female'), 5 is quality ('good' or 'bad'), 6 is test
    TARGET_INDEXES=[1, 3, 4, 5],

    CODES=['100', '101', '102', '200', '201', '202']
))


#Result configuration for result generation after evaluation is done
RESULT_CONFIG = Bunch(dict(
    # Error margin on various ranges
    ACCURACIES=[0.04, 0.1, 0.21, 0.42],  # 40gms, 100gms, 210gms, 420gms
    ACCURACY_MAIN_THRESH=0.21,  # 210gms
    AGE_BUCKETS=[0, 1, 2, 3, 4, 5],

    COLUMNS=['qrcode', 'artifact', 'scantype', 'GT', 'predicted'],

    # uncertainty
    USE_UNCERTAINTY=False,  # Flag to enable model uncertainty calculation

    # path of csv file in the experiment which final result is stored
    SAVE_PATH=f'./outputs/{CONFIG_NAME}',
))
