
import sys
from pathlib import Path

from azureml.core import Environment

sys.path.append(str(Path(__file__).parents[1] / 'endpoints'))  # noqa
from constants import REPO_DIR  # noqa: E402


def cgm_environment(workspace, curated_env_name, env_exist):
    if env_exist:
        cgm_env = Environment.get(workspace=workspace, name=curated_env_name)
    else:
        cgm_env = Environment.from_conda_specification(
            name=curated_env_name, file_path=REPO_DIR / "environment_train.yml")
        cgm_env.docker.enabled = True
        cgm_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'
    return cgm_env
