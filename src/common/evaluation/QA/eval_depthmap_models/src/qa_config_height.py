from bunch import Bunch

# Details of model used for evaluation
MODEL_CONFIG = Bunch(dict(
    EXPERIMENT_NAME='q3-depthmap-plaincnn-height-95k',
    RUN_ID='q3-depthmap-plaincnn-height-95k_1610709896_ef7f755d',  # Run 5

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

#Details of Evaluation Dataset
DATA_CONFIG = Bunch(dict(
    NAME='anon-realtime-testdata',  # Name of evaluation dataset

    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,

    BATCH_SIZE=512,  # Batch size for evaluation
    NORMALIZATION_VALUE=7.5,

    TARGET_INDEXES=[0],  # 0 is height, 1 is weight.
    CODES=['100', '101', '102', '200', '201', '202']
))


# Result configuration for result generation after evaluation is done
RESULT_CONFIG = Bunch(dict(
    # Error margin on various ranges
    #EVALUATION_ACCURACIES = [.2, .4, .8, 1.2, 2., 2.5, 3., 4., 5., 6.]
    ACCURACIES=[.2, .4, .6, 1, 1.2, 2., 2.5, 3., 4., 5., 6.],  # 0.2cm, 0.4cm, 0.6cm, 1cm, ...
    COLUMNS=['qrcode', 'artifact', 'scantype', 'GT', 'predicted'],

    # path of csv file in the experiment which final result is stored
    SAVE_PATH='./outputs/result.csv',
))
