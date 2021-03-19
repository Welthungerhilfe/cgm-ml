from bunch import Bunch

# Error margin on various ranges
EVALUATION_ACCURACIES = [.2, .4, .6, 1., 1.2, 2., 2.5, 3., 4., 5., 6.]
MODEL_CKPT_FILENAME = "best_model.ckpt"

DAYS_IN_YEAR = 365

HEIGHT_IDX = 0
WEIGHT_IDX = 1
MUAC_IDX = 2
AGE_IDX = 3
SEX_IDX = 4
GOODBAD_IDX = 5

SEX_DICT = {'female': 0., 'male': 1.}
GOODBAD_DICT = {'bad': 0., 'good': 1., 'delete': 2.}

COLUMN_NAME_AGE = 'GT_age'
COLUMN_NAME_SEX = 'GT_sex'
COLUMN_NAME_GOODBAD = 'GT_goodbad'
CODE_TO_SCANTYPE = {
    '100': '_standingfront',
    '101': '_standing360',
    '102': '_standingback',
    '200': '_lyingfront',
    '201': '_lyingrot',
    '202': '_lyingback',
}

DATA_AUGMENTATION_SAME_PER_CHANNEL = "same_per_channel"
DATA_AUGMENTATION_DIFFERENT_EACH_CHANNEL = "different_each_channel"
DATA_AUGMENTATION_NO = "no"

SAMPLING_STRATEGY_SYSTEMATIC = "systematic"
SAMPLING_STRATEGY_WINDOW = "window"

CONFIG = Bunch(dict(
    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,
    NORMALIZATION_VALUE=7.5,
    TARGET_INDEXES=[0],  # 0 is height, 1 is weight.
    DATA_AUGMENTATION_MODE=DATA_AUGMENTATION_NO,
    SAMPLING_STRATEGY=SAMPLING_STRATEGY_SYSTEMATIC,
    N_ARTIFACTS=5,
    N_REPEAT_DATASET=1,
    CODES_FOR_POSE_AND_SCANSTEP=('100', '101', '102', '200', '201', '202'),
))
