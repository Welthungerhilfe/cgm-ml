from pathlib import Path

REPO_DIR = Path(__file__).parents[6].absolute()
DATA_DIR_ONLINE_RUN = Path("/tmp/data/")
MODEL_CKPT_FILENAME = "best_model.ckpt"

PIP_PACKAGES = [
    "azureml-dataprep[fuse,pandas]",
    "glob2",
    "opencv-python==4.1.2.30",
    "matplotlib",
    "tensorflow-addons==0.11.2",
    "bunch==1.0.1",
    "wandb",
]
