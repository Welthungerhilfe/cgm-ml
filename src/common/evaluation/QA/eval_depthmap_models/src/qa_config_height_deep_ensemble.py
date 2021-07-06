import os

from bunch import Bunch

CONFIG_NAME = os.path.splitext(os.path.basename(__file__))[0]

# Details of model used for evaluation
MODEL_CONFIG = Bunch(dict(
    EXPERIMENT_NAME='2021q1-depthmap-ensemble-height-95k',
    RUN_IDS=['2021q1-depthmap-ensemble-height-95k_1622230917_290d733e',
             '2021q1-depthmap-ensemble-height-95k_1622230877_02b7cc34',
             '2021q1-depthmap-ensemble-height-95k_1622230838_187c2c8f',
             '2021q1-depthmap-ensemble-height-95k_1622230815_3a6833d4',
             '2021q1-depthmap-ensemble-height-95k_1622230796_d18037bb',
             '2021q1-depthmap-ensemble-height-95k_1622230770_32a7a89c',
             '2021q1-depthmap-ensemble-height-95k_1622230729_c239a326',
             '2021q1-depthmap-ensemble-height-95k_1622230699_886a2a9d',
             '2021q1-depthmap-ensemble-height-95k_1622230664_d9d108b4',
             '2021q1-depthmap-ensemble-height-95k_1622230624_f791db0c',
             '2021q1-depthmap-ensemble-height-95k_1622230551_76ff056b',
             '2021q1-depthmap-ensemble-height-95k_1622230426_12111939',
             '2021q1-depthmap-ensemble-height-95k_1622230397_a2f46051',
             '2021q1-depthmap-ensemble-height-95k_1622230371_7b4cc0a2',
             '2021q1-depthmap-ensemble-height-95k_1622230334_67c64a77'],
    INPUT_LOCATION='outputs',
    NAME='best_model.ckpt',
))


EVAL_CONFIG = Bunch(dict(
    # Name of evaluation
    NAME='2021q1-depthmap-ensemble-height-95k',

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
    # EVALUATION_ACCURACIES = [.2, .4, .8, 1.2, 2., 2.5, 3., 4., 5., 6.]
    ACCURACIES=[.2, .4, .6, 1, 1.2, 2., 2.5, 3., 4., 5., 6.],  # 0.2cm, 0.4cm, 0.6cm, 1cm, ...
    ACCURACY_MAIN_THRESH=1.0,
    COLUMNS=['qrcode', 'artifact', 'scantype', 'GT', 'predicted'],

    USE_UNCERTAINTY=True,  # Flag to enable model uncertainty calculation
    UNCERTAINTY_THRESHOLD_IN_CM=4.,

    TARGET_INDEXES=[0],
    # path of csv file in the experiment which final result is stored
    SAVE_PATH=f'./outputs/{CONFIG_NAME}',
))
