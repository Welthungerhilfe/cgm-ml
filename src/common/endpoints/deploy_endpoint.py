import sys
from pathlib import Path

from azureml.core import Workspace
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice  # noqa: E401
from azureml.core.webservice import LocalWebservice
from config_deepensemble_1 import CONFIG

from constants import REPO_DIR

sys.path.append(str(Path(__file__).parents[1] / 'model_utils'))   # noqa
import environment  # noqa: E402

if __name__ == "__main__":

    workspace = Workspace.from_config()
    model = Model(workspace, name=CONFIG.MODEL_NAME)

    cgm_env = environment.cgm_environment(workspace=workspace, curated_env_name="cgm-env", env_exist=True)

    inference_config_aci = InferenceConfig(
        environment=cgm_env,
        entry_script=str(REPO_DIR / "src/common/endpoints/entry_script_aci.py"),
    )

    if CONFIG.LOCALTEST:
        deployment_config = LocalWebservice.deploy_configuration(port=6789)
    else:
        deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=4)

    service = Model.deploy(workspace, CONFIG.ENDPOINT_NAME, [model],
                           inference_config_aci, deployment_config, overwrite=True,)
    service.wait_for_deployment(show_output=True)
    print(service.swagger_uri)
