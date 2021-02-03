from pathlib import Path

REPO_DIR = Path(__file__).parents[6].absolute()

PIP_PACKAGES = [
    "azureml-dataprep[fuse,pandas]",
    "glob2",
    "opencv-python==4.1.2.30",
    "matplotlib",
    "imgaug==0.4.0",
    "tensorflow-addons==0.12.1",
    "bunch==1.0.1",
]
