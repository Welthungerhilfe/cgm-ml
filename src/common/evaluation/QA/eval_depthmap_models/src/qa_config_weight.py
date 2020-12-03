class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


#Details of model used for evaluation
MODEL_CONFIG = dotdict(dict(
    EXPERIMENT_NAME='q4-depthmap-plaincnn-weight-95k',
    RUN_ID='q4-depthmap-plaincnn-weight-95k_1605774694_c216f6c5',
    INPUT_LOCATION='outputs',
    NAME='best_model.ckpt',
))


EVAL_CONFIG = dotdict(dict(
    #Name of evaluation
    NAME='q4-depthmap-plaincnn-weight-95k-run_12',

    #Experiment in Azure ML which will be used for evaluation
    EXPERIMENT_NAME="QA-pipeline",
    CLUSTER_NAME="gpu-cluster",

    #Used for Debug the QA pipeline
    DEBUG_RUN=False,

    #Will run eval on specified # of scan instead of full dataset
    DEBUG_NUMBER_OF_SCAN=50,

    SPLIT_SEED=0,
))

#Details of Evaluation Dataset
DATA_CONFIG = dotdict(dict(
    #Name of evaluation dataset
    NAME='anon-depthmap-testset',

    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,

    #Batch size for evaluation
    BATCH_SIZE=256,
    NORMALIZATION_VALUE=7.5,

    # Parameters for dataset generation.
    TARGET_INDEXES=[1],  # 0 is height, 1 is weight.

    CODE_TO_SCANTYPE={
        '100': '_front',
        '101': '_360',
        '102': '_back',
        '200': '_lyingfront',
        '201': '_lyingrot',
        '202': '_lyingback',
    }
))


#Result configuration for result generation after evaluation is done
RESULT_CONFIG = dotdict(dict(
    # Error margin on various ranges
    ACCURACIES=[0.04, 0.1, 0.21, 0.42],  # 40 gms, 100 gms, 210 gms, 420 gms

    COLUMNS=['qrcode', 'artifact', 'scantype', 'GT', 'predicted'],

    #path of csv file in the experiment which final result is stored
    SAVE_PATH='./outputs/result.csv',
))
