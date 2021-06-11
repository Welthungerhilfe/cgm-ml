
from azureml.core import Environment, Workspace
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice, LocalWebservice

from config import CONFIG
from constants import REPO_DIR

workspace = Workspace.from_config()
model = Model(workspace, name=CONFIG.MODEL_NAME)

curated_env_name = "cgm-env"

ENV_EXISTS = True
if ENV_EXISTS:
    cgm_env = Environment.get(workspace=workspace, name=curated_env_name)
else:
    cgm_env = Environment.from_conda_specification(name=curated_env_name, file_path=REPO_DIR / "environment_train.yml")
    cgm_env.docker.enabled = True
    cgm_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'
    # cgm_env.register(workspace)  # Please be careful not to overwrite existing environments

inference_config_aci = InferenceConfig(
    environment=cgm_env,
    entry_script=str(REPO_DIR / "src/common/endpoints/entry_script_aci.py"),
)

#deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=4)

deployment_config = LocalWebservice.deploy_configuration(port=6789)
service = Model.deploy(workspace, CONFIG.ENDPOINT_NAME, [model],
                       inference_config_aci, deployment_config, overwrite=True,)
service.wait_for_deployment(show_output=True)
print(service.swagger_uri)
