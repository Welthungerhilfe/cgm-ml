from bunch import Bunch

#Details of model used for evaluation
MODEL_CONFIG = Bunch(dict(
    EXPERIMENT_NAME='q3-depthmap-plaincnn-height-95k',

    # RUN_ID='q3-depthmap-plaincnn-height-95k_1597988908_42c4ef33',  # Run 3
    RUN_ID='q3-depthmap-plaincnn-height-95k_1600451633_cb44f6db',  # Run 17

    INPUT_LOCATION='outputs',
    NAME='best_model.h5',
))


EVAL_CONFIG = Bunch(dict(
    #Name of evaluation
    NAME='q3-depthmap-plaincnn-height-95k_run_04',

    #Experiment in Azure ML which will be used for evaluation
    EXPERIMENT_NAME="QA-pipeline",
    CLUSTER_NAME="gpu-cluster",

    #Used for Debug the QA pipeline
    DEBUG_RUN=False,

    #Will run eval on specified # of scan instead of full dataset
    DEBUG_NUMBER_OF_SCAN=200,

    SPLIT_SEED=0,
))

#Details of Evaluation Dataset
DATA_CONFIG = Bunch(dict(
    #Name of evaluation dataset
    NAME='anon-realtime-testdata',

    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,

    #Batch size for evaluation
    BATCH_SIZE=512,
    NORMALIZATION_VALUE=7.5,

    # Parameters for dataset generation.
    TARGET_INDEXES=[0, 3, 4, 5],  # 0 is height, 1 is weight, 2 is muac, 3 is age, 4 is sex('male' or 'female'), 5 is quality ('good' or 'bad'), 6 is test

    CODES=['100', '101', '102', '200', '201', '202']
))


#Result configuration for result generation after evaluation is done
RESULT_CONFIG = Bunch(dict(
    # Error margin on various ranges
    #EVALUATION_ACCURACIES = [.2, .4, .8, 1.2, 2., 2.5, 3., 4., 5., 6.]
    ACCURACIES=[.2, .4, .6, 1., 1.2, 2., 2.5, 3., 4., 5., 6.],  # 0.2cm, 0.4cm, 0.6cm, 1cm, ...
    ACCURACY_MAIN_HEIGHT_THRESH=1.0,  # 1cm
    AGE_BUCKETS=[0, 1, 2, 3, 4, 5],

    COLUMNS=['qrcode', 'artifact', 'scantype', 'GT', 'predicted'],

    # uncertainty
    USE_UNCERTAINTY=True,  # Flag to enable model uncertainty calculation
    NUM_DROPOUT_PREDICTIONS=16,
    DROPOUT_STRENGTH=1,  # 1.0 means like original model

    #path of csv file in the experiment which final result is stored
    SAVE_PATH='./outputs/height',
))
