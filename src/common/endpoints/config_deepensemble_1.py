from bunch import Bunch

from constants import REPO_DIR

CONFIG = Bunch(dict(
    MODEL_NAME="2021q1-depthmap-ensemble-height-95k",
    VERSION='1',
    ENDPOINT_NAME='aci-21q1-ensemble-h-95k-m01-v1',
    EXPERIMENT_NAME='2021q1-depthmap-ensemble-height-95k',
    RUN_ID='2021q1-depthmap-ensemble-height-95k_1622230334_67c64a77',
    LOCALTEST=False,
    TEST_FILES=[REPO_DIR / 'src/common/depthmap_toolkit/tests/static_files/4ed427b5-3fd9-4f4d-8e58-19e39c7d77b6',
                REPO_DIR / 'src/common/depthmap_toolkit/tests/static_files/4ed427b5-3fd9-4f4d-8e58-19e39c7d77b6']
))
