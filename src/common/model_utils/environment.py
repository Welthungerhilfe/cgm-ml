
import sys

from azureml.core import Environment

sys.path.append('./src/common/endpoints')  # noqa
from constants import REPO_DIR


def cgm_environemnt(workspace, curated_env_name, env_exist):
    if env_exist:
        cgm_env = Environment.get(workspace=workspace, name=curated_env_name)
    else:
        cgm_env = Environment.from_conda_specification(
            name=curated_env_name, file_path=REPO_DIR / "environment_train.yml")
        cgm_env.docker.enabled = True
        cgm_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'
    return cgm_env
