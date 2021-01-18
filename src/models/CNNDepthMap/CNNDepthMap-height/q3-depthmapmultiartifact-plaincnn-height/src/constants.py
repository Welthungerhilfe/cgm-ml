from pathlib import Path
from src.models.CNNDepthMap.CNNDepthMap-height.q3-depthmap-plaincnn-height.src.constants import (
    PIP_PACKAGES)

REPO_DIR = Path(__file__).parents[6].absolute()

PIP_PACKAGES = [
    "azureml-dataprep[fuse,pandas]",
    "glob2",
    "opencv-python==4.1.2.30",
    "matplotlib",
    "imgaug==0.4.0",
    "tensorflow-addons==0.11.2",
    "bunch==1.0.1",
]
