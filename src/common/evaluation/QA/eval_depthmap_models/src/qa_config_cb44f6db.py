from bunch import Bunch

# Details of model used for evaluation
MODEL_CONFIG = Bunch(dict(
    EXPERIMENT_NAME='q3-depthmap-plaincnn-height-95k',

    # RUN_ID='q3-depthmap-plaincnn-height-95k_1597988908_42c4ef33',  # Run 3
    RUN_ID='q3-depthmap-plaincnn-height-95k_1600451633_cb44f6db',  # Run 17

    INPUT_LOCATION='outputs',
    NAME='best_model.h5',
))


EVAL_CONFIG = Bunch(dict(
    # Name of evaluation
    NAME='q3-depthmap-plaincnn-height-95k_run_04',

    # Experiment in Azure ML which will be used for evaluation
    EXPERIMENT_NAME="QA-pipeline",
    CLUSTER_NAME="gpu-cluster",

    # Used for Debug the QA pipeline
    DEBUG_RUN=True,

    # Will run eval on specified # of scan instead of full dataset
    DEBUG_NUMBER_OF_SCAN=5,

    SPLIT_SEED=0,
))

FILTER_CONFIG = Bunch(dict(
    IS_ENABLED=True,  # 'False'
    EXPERIMENT_NAME='q4-rgb-plaincnn-classifaction-standing-lying-8k',

    RUN_ID='q4-rgb-plaincnn-classifaction-standing-lying-8k_1602316038_3ebdb326',  # Run 3
    # RUN_ID = 'q3-depthmap-plaincnn-height-95k_1600451633_cb44f6db',     #Run 17

    INPUT_LOCATION='outputs',
    NAME='best_filter.h5',
))

# Details of Evaluation Dataset
DATA_CONFIG = Bunch(dict(
    # Name of evaluation dataset
    NAME='anon-depthmap-rgb-timestamp',

    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,

    # Batch size for evaluation
    BATCH_SIZE=512,
    NORMALIZATION_VALUE=7.5,

    # Parameters for dataset generation.
    TARGET_INDEXES=[0, 3],  # 0 is height, 1 is weight.

    CODE_TO_SCANTYPE={
        '100': '_front',
        '101': '_360',
        '102': '_back',
        '200': '_lyingfront',
        '201': '_lyingrot',
        '202': '_lyingback',
    }
))


# Result configuration for result generation after evaluation is done
RESULT_CONFIG = Bunch(dict(
    # Error margin on various ranges
    #EVALUATION_ACCURACIES = [.2, .4, .8, 1.2, 2., 2.5, 3., 4., 5., 6.]
    ACCURACIES=[.2, .4, .6, 1, 1.2, 2., 2.5, 3., 4., 5., 6.],  # 0.2cm, 0.4cm, 0.6cm, 1cm, ...
    AGE_BUCKETS=[0, 1, 2, 3, 4, 5],

    COLUMNS=['qrcode', 'artifact', 'scantype', 'GT', 'predicted'],

    # path of csv file in the experiment which final result is stored
    SAVE_PATH='./outputs/height',
))
