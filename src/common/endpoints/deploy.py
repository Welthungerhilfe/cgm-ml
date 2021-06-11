
from azureml.core import Environment, Workspace
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice, LocalWebservice

from config import CONFIG
from constants import REPO_DIR

ws = Workspace.from_config()
model = Model(ws, name=CONFIG.MODEL_NAME)

cgm_env = Environment.from_conda_specification(
    name='project_environment', file_path=REPO_DIR / "environment_train.yml")
cgm_env.docker.enabled = True
cgm_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'
inference_config_aci = InferenceConfig(
    environment=cgm_env,
    entry_script=str(REPO_DIR / "src/common/endpoints/entry_script_aci.py"),
)


deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=4)

#deployment_config = LocalWebservice.deploy_configuration(port=6789)
service = Model.deploy(ws, CONFIG.ENDPOINT_NAME, [model], inference_config_aci, deployment_config, overwrite=True,)
service.wait_for_deployment(show_output=True)
print(service.swagger_uri)
