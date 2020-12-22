from bunch import Bunch

#Details of model used for evaluation
MODEL_CONFIG = Bunch(dict(
    EXPERIMENT_NAME='q3-depthmap-plaincnn-height-95k',

    RUN_ID='q3-depthmap-plaincnn-height-95k_1597988908_42c4ef33',  # Run 3
    #RUN_ID = 'q3-depthmap-plaincnn-height-95k_1600451633_cb44f6db',     #Run 17

    INPUT_LOCATION='outputs',
    NAME='best_model.h5',
))


EVAL_CONFIG = Bunch(dict(
    #Name of evaluation
    NAME='eval-standadisation-test',

    #Experiment in Azure ML which will be used for evaluation
    EXPERIMENT_NAME="QA-pipeline",
    CLUSTER_NAME="gpu-cluster",

    #Used for Debug the QA pipeline
    DEBUG_RUN=False,
    #DEBUG_RUN = True,

    DEBUG_LOG=True,

    TEM_THRESHOLD=0.6,

    #Will run eval on specified # of scan instead of full dataset
    DEBUG_NUMBER_OF_DEPTHMAP=500,

    SPLIT_SEED=0,
))

#Details of Evaluation Dataset
DATA_CONFIG = Bunch(dict(
    #Name of evaluation dataset
    NAME='standardization_test_01',

    #Excel file in which measurement of each enumerator is stored
    #while standardisation test
    EXCEL_NAME='Standardization test October.xlsx',
    SHEET_NAME='DRS',

    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,

    #Batch size for evaluation
    BATCH_SIZE=256,
    NORMALIZATION_VALUE=7.5,

    # Parameters for dataset generation.
    TARGET_INDEXES=[0],  # 0 is height, 1 is weight.

    CODES=['100', '101', '102', '200', '201', '202']
))


#Configuration for report generation after evaluation is done
RESULT_CONFIG = Bunch(dict(
    # Error margin on various ranges
    #EVALUATION_ACCURACIES = [.2, .4, .8, 1.2, 2., 2.5, 3., 4., 5., 6.]
    ACCURACIES=[.2, .4, .6, 1, 1.2, 2., 2.5, 3., 4., 5., 6.],
    COLUMNS=['qrcode', 'artifact', 'scantype', 'GT', 'predicted'],

    #path of csv file in the experiment which final result is stored
    SAVE_PATH='./outputs/result.csv',
    SAVE_TEM_PATH='./outputs/result_tem.csv',

))
